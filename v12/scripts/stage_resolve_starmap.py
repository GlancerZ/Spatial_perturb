"""Stage evidenced Resolve RNA and audit STARmap archives on a compute node.

No nuclear annotations are inferred. AppleDouble ._goodPoints.mat files are
metadata resource forks, not molecule coordinates. Unsupported STARmap payloads
are recorded as blocked instead of synthesizing point coordinates from images.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_unlabeled_subcellular import _start, _finish, _save, _number, _sha256


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def read_sample_metadata(path=None, geo_soft=None):
    """Only explicit sample metadata can assign species; no filename guessing."""
    if path is not None and geo_soft is not None:
        raise ValueError("Provide sample metadata JSON or GEO SOFT, not both")
    if path is not None:
        result = json.loads(Path(path).read_text())
        for gsm, record in result.items():
            if not re.fullmatch(r"GSM\d+", gsm) or not isinstance(record, dict):
                raise ValueError("Sample metadata must map GSM accessions to objects")
            if record.get("species") not in ("Mus musculus", "Homo sapiens") or not record.get("evidence"):
                raise ValueError("Sample species needs explicit mouse/human identity and evidence")
        return result
    if geo_soft is None:
        return {}
    result, current = {}, None
    with Path(geo_soft).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("^SAMPLE = "):
                current = line.partition(" = ")[2].strip()
                if not re.fullmatch(r"GSM\d+", current):
                    raise ValueError("Invalid GEO SAMPLE accession")
                result.setdefault(current, {"evidence": str(Path(geo_soft).resolve())})
            elif current is not None and line.startswith("!Sample_organism_ch1 = "):
                species = line.partition(" = ")[2].strip()
                if species not in ("Mus musculus", "Homo sapiens"):
                    raise ValueError(f"Unsupported or ambiguous GEO species: {species}")
                if result[current].get("species", species) != species:
                    raise ValueError(f"Conflicting species for {current}")
                result[current]["species"] = species
    return {gsm: record for gsm, record in result.items() if "species" in record}


def _bounded_lines(stream, limit, budget=None):
    total = 0
    while True:
        line = stream.readline(min(2 * 1024**2, limit - total + 1))
        if not line:
            return
        total += len(line.encode("utf-8"))
        if budget is not None:
            budget["remaining"] -= len(line.encode("utf-8"))
            if budget["remaining"] < 0:
                raise ValueError("Total decompressed RNA byte budget exceeded")
        if total > limit or len(line) >= 2 * 1024**2:
            raise ValueError("Decompressed RNA member or record exceeds byte cap")
        yield line


def _table_rows(stream, columns, delimiter, has_header, byte_cap, *, require_cell_id=True, budget=None):
    if len(delimiter) != 1:
        raise ValueError("Delimiter must be exactly one character")
    required = {"x", "y", "gene"} | ({"cell_id"} if require_cell_id else set())
    if not required.issubset(columns) or set(columns) - (required | {"z"}):
        raise ValueError("Explicit Resolve columns must map x,y,gene,cell_id and optional z")
    reader = csv.reader(_bounded_lines(stream, byte_cap, budget), delimiter=delimiter)
    if has_header:
        header = next(reader, None)
        if not header or len(header) != len(set(header)):
            raise ValueError("Missing or duplicated source RNA header")
        if any(not isinstance(value, str) or value not in header for value in columns.values()):
            raise ValueError(f"Mapped RNA columns not found in actual header: {header}")
        mapping = {name: header.index(column) for name, column in columns.items()}
        width = len(header)
    else:
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in columns.values()):
            raise ValueError("Headerless mapping must use nonnegative zero-based column indices")
        mapping, width = columns, None
    for row_index, fields in enumerate(reader):
        if not fields:
            continue
        if (width is not None and len(fields) != width) or len(fields) <= max(mapping.values()):
            raise ValueError(f"Malformed RNA row {row_index}")
        yield row_index, {name: fields[index] for name, index in mapping.items()}


REFERENCE_COUNTERS = ("differing_pair_count", "coordinate_diff", "gene_diff", "original_extra_rows", "assigned_extra_rows")


def _validate_reference_fields(fields):
    if fields is None:
        return
    if not fields["gene"].strip() or fields["gene"].strip() != fields["gene"]:
        raise ValueError("Missing or whitespace-ambiguous source gene label")
    for axis in ("x", "y", "z"):
        if axis in fields and not (axis == "z" and fields[axis] == ""):
            _number(fields[axis], axis)


class _ReferenceComparison:
    def __init__(self, mode, aliases, gsm, alias_counts):
        self.mode, self.aliases, self.gsm, self.alias_counts = mode, aliases, gsm, alias_counts
        self.stats = {key: 0 for key in REFERENCE_COUNTERS}
        self.stats.update(first_differences=[], explicit_gene_alias_rows=0, matches_with_explicit_aliases=True)

    def compare(self, original, assigned, row_index):
        _validate_reference_fields(original)
        _validate_reference_fields(assigned)
        if original == assigned:
            return
        allowed = False
        if original is None:
            self.stats["assigned_extra_rows"] += 1
        elif assigned is None:
            self.stats["original_extra_rows"] += 1
        else:
            self.stats["differing_pair_count"] += 1
            coordinates_equal = {key: value for key, value in original.items() if key != "gene"} == {key: value for key, value in assigned.items() if key != "gene"}
            genes_equal = original["gene"] == assigned["gene"]
            self.stats["coordinate_diff"] += int(not coordinates_equal)
            self.stats["gene_diff"] += int(not genes_equal)
            allowed = coordinates_equal and self.aliases.get(original["gene"]) == assigned["gene"]
            if allowed:
                self.alias_counts[original["gene"]] += 1
                self.stats["explicit_gene_alias_rows"] += 1
        if len(self.stats["first_differences"]) < 10:
            self.stats["first_differences"].append({"sample_id": self.gsm, "row_index": row_index,
                "original": original, "assigned": assigned, "explicit_alias_match": allowed})
        if not allowed:
            self.stats["matches_with_explicit_aliases"] = False
            if self.mode == "strict":
                raise ValueError(f"Assigned/original RNA table mismatch at {self.gsm} row {row_index}: original={original}, assigned={assigned}")


def stage_resolve(input_tar, output, *, columns, delimiter="\t", has_header=True,
                  sample_metadata=None, geo_soft=None, expected_sha256=None,
                  coordinate_unit="unverified_source_units", coordinate_evidence=None,
                  rows_per_shard=100000, max_member_bytes=512 * 1024**2,
                  max_total_bytes=8 * 1024**3, expected_samples=None,
                  unassigned_cell_ids=("NotAssigned", "0", "-1", ""),
                  original_gene_aliases=None, reference_validation_mode="strict"):
    if reference_validation_mode not in ("strict", "report"):
        raise ValueError("Reference validation mode must be strict or report")
    if max_member_bytes <= 0 or max_total_bytes <= 0:
        raise ValueError("Positive archive byte caps are required")
    if coordinate_unit not in ("unverified_source_units", "micron", "nanometer", "pixel"):
        raise ValueError("Unsupported coordinate unit")
    if coordinate_unit != "unverified_source_units" and not coordinate_evidence:
        raise ValueError("Physical coordinate units require source evidence")
    aliases = {} if original_gene_aliases is None else original_gene_aliases
    if not isinstance(aliases, dict) or any(not isinstance(original, str) or not original or
            not isinstance(assigned, str) or not assigned for original, assigned in aliases.items()):
        raise ValueError("original_gene_aliases must explicitly map nonempty original names to assigned names")
    metadata = read_sample_metadata(sample_metadata, geo_soft)
    destination, manifest, writer = _start(input_tar, output, "resolve_mc_gse325911", expected_sha256, rows_per_shard)
    manifest.update(platform="Resolve Molecular Cartography", species="mixed_or_unverified_per_sample",
        source_variant="author assigned molecule table", reference_validation_mode=reference_validation_mode,
        coordinate_unit=coordinate_unit, coordinate_evidence=coordinate_evidence,
        coordinate_frame="source_sample_local_unregistered", coordinate_transform="identity",
        panel_complete=False, measured_panel=None, observed_genes="observed_genes.parquet",
        cell_assignment_semantics="Publisher assignment to dilated DAPI masks; this is not nuclear/cytoplasmic classification",
        known_blockers=["No verified nuclear boundaries or per-RNA compartment labels",
                        "Dilated DAPI cell assignment cannot establish genuine cytoplasm",
                        "Full assayed panel must be resolved from each GSM/GPL; observed genes are not a complete panel",
                        "Sample-local coordinate units and inter-sample registration remain source-specific"],
        provenance={"source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325911",
                    "selection": "Only *_results_assigned.txt.gz; *_results.txt.gz is not added again",
                    "column_mapping": columns, "header_present": has_header, "delimiter": delimiter,
                    "original_gene_aliases": aliases,
                    "original_gene_aliases_sha256": hashlib.sha256(json.dumps(aliases, sort_keys=True).encode()).hexdigest(),
                    "gene_alias_policy": "Strict mode permits only explicit exact aliases with identical coordinates; report mode records all differences without transforming assigned output",
                    "reference_comparison_semantics": "Same-position textual row comparison; ordering differences can produce many paired mismatches. No reordering, molecule-set equivalence claim, or original/assigned merge."})
    if reference_validation_mode == "report" and manifest.get("training_ready") is not False:
        raise ValueError("Report mode is limited to explicitly non-training intermediate data")
    if sample_metadata or geo_soft:
        metadata_path = Path(sample_metadata or geo_soft)
        manifest["sample_metadata_source"] = {"path": str(metadata_path.resolve()), "sha256": _sha256(metadata_path)}
    try:
        observed, samples, seen_names, seen_samples, total_declared = {}, [], set(), set(), 0
        alias_counts = {name: 0 for name in aliases}
        budget = {"remaining": max_total_bytes}
        with tarfile.open(input_tar, mode="r:*") as archive:
            for member in archive:
                path = _safe_name(member.name)
                if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                    raise ValueError(f"Unsafe archive member type: {member.name}")
                if member.isdir():
                    continue
                if member.name in seen_names:
                    raise ValueError("Duplicate archive member name")
                seen_names.add(member.name)
                total_declared += member.size
                if member.size < 0 or total_declared > max_total_bytes:
                    raise ValueError("Archive exceeds total byte budget")
                if not path.name.endswith("_results_assigned.txt.gz"):
                    continue
                match = re.match(r"(GSM\d+)_", path.name)
                if not match or match.group(1) in seen_samples:
                    raise ValueError("Assigned RNA files require exactly one table per GSM")
                gsm = match.group(1)
                seen_samples.add(gsm)
                if member.size > max_member_bytes:
                    raise ValueError("Compressed RNA member exceeds byte cap")
                raw = archive.extractfile(member)
                if raw is None:
                    raise ValueError("Missing RNA member payload")
                counterpart_name = member.name.replace("_results_assigned.txt.gz", "_results.txt.gz")
                try:
                    counterpart = archive.getmember(counterpart_name)
                except KeyError as error:
                    raise ValueError(f"Missing original RNA table paired to assigned table: {counterpart_name}") from error
                _safe_name(counterpart.name)
                if not counterpart.isfile() or counterpart.size > max_member_bytes:
                    raise ValueError("Unsafe or oversized original RNA counterpart")
                original_raw = archive.extractfile(counterpart)
                if original_raw is None:
                    raise ValueError("Cannot read original RNA counterpart")
                count, assigned_count, genes = 0, 0, set()
                comparison = _ReferenceComparison(reference_validation_mode, aliases, gsm, alias_counts)
                with raw, original_raw, gzip.GzipFile(fileobj=raw) as decoded, gzip.GzipFile(fileobj=original_raw) as original_decoded, io.TextIOWrapper(decoded, encoding="utf-8-sig", newline="") as text, io.TextIOWrapper(original_decoded, encoding="utf-8-sig", newline="") as original_text:
                    original_columns = {name: index for name, index in columns.items() if name != "cell_id"}
                    original_rows = _table_rows(original_text, original_columns, delimiter, has_header, max_member_bytes, require_cell_id=False, budget=budget)
                    for row_index, row in _table_rows(text, columns, delimiter, has_header, max_member_bytes, budget=budget):
                        original_row = next(original_rows, None)
                        assigned_fields = {name: value for name, value in row.items() if name != "cell_id"}
                        comparison.compare(original_row[1] if original_row is not None else None, assigned_fields, row_index)
                        gene = row["gene"]
                        if not gene.strip() or gene.strip() != gene:
                            raise ValueError("Missing or whitespace-ambiguous source gene label")
                        original_cell = row["cell_id"]
                        cell = None if original_cell in unassigned_cell_ids else f"{gsm}:{original_cell}"
                        writer.add(dataset_id=manifest["dataset_id"], sample_id=gsm, fov=gsm, cell_id=cell,
                            gene=gene, x=_number(row["x"], "x"), y=_number(row["y"], "y"),
                            z=_number(row["z"], "z") if "z" in row and row["z"] != "" else None,
                            coordinate_unit=coordinate_unit, z_unit=coordinate_unit if "z" in row else None,
                            compartment="unknown", auxiliary_label=None, source_member=member.name, source_record=row_index)
                        count += 1
                        assigned_count += int(cell is not None)
                        genes.add(gene)
                    for original_index, original_fields in original_rows:
                        comparison.compare(original_fields, None, original_index)
                samples.append({"sample_id": gsm, "source_member": member.name, "rna_rows": count,
                    "assigned_rows": assigned_count, "unassigned_rows": count - assigned_count,
                    "species": metadata.get(gsm, {}).get("species"),
                    "species_evidence": metadata.get(gsm, {}).get("evidence"),
                    "species_status": "verified_metadata" if gsm in metadata else "unverified_not_inferred_from_name",
                    "original_assigned_rows_identical": not any(comparison.stats[key] for key in REFERENCE_COUNTERS),
                    "original_assigned_rows_match_with_explicit_gene_aliases": comparison.stats["matches_with_explicit_aliases"],
                    "explicit_gene_alias_rows": comparison.stats["explicit_gene_alias_rows"],
                    "reference_comparison": comparison.stats, "original_member": counterpart_name})
                observed[gsm] = genes
                manifest["members"].append({"name": member.name, "compressed_bytes": member.size, "rna_rows": count})
        if expected_samples is not None and len(samples) != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, found {len(samples)}")
        records = [{"sample_id": gsm, "species": metadata.get(gsm, {}).get("species"), "gene": gene,
                    "observed": True, "assayed": None} for gsm, genes in observed.items() for gene in sorted(genes)]
        observed_schema = pa.schema([("sample_id", pa.string()), ("species", pa.string()), ("gene", pa.string()),
                                    ("observed", pa.bool_()), ("assayed", pa.bool_())])
        pq.write_table(pa.Table.from_pylist(records, schema=observed_schema), destination / "observed_genes.parquet", compression="zstd")
        manifest.update(samples=samples, sample_count=len(samples),
                        unknown_species_samples=sum(sample["species"] is None for sample in samples),
                        measured_gene_count=None, observed_sample_gene_pairs=len(records),
                        decompressed_rna_bytes_checked=max_total_bytes - budget["remaining"],
                        explicit_gene_alias_rows=sum(alias_counts.values()),
                        gene_alias_matches={name: {"assigned_gene": aliases[name], "rows": count} for name, count in alias_counts.items()})
        reference_summary = {key: sum(sample["reference_comparison"][key] for sample in samples) for key in REFERENCE_COUNTERS}
        reference_summary["first_differences"] = [example for sample in samples for example in sample["reference_comparison"]["first_differences"]][:10]
        not_identical = any(reference_summary[key] for key in REFERENCE_COUNTERS)
        manifest.update(reference_comparison=reference_summary, original_assigned_not_identical=not_identical,
                        reference_validation_status=("reported_not_identical" if not_identical else "reported_identical") if reference_validation_mode == "report" else ("passed_with_explicit_gene_aliases" if not_identical else "passed_exact"))
        if reference_validation_mode == "report" and not_identical:
            manifest["known_blockers"].append("Original and author-assigned RNA tables are not identical; assigned variant retained independently, reference consistency not passed")
        return _finish(destination, manifest, writer)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(destination, manifest)
        raise


def audit_starmap(input_zip, output, *, expected_sha256=None):
    destination, manifest, _ = _start(input_zip, output, "starmap_plus_64_zenodo_7332091", expected_sha256, 100000)
    manifest.update(platform="STARmap PLUS", species="Mus musculus", species_evidence="https://zenodo.org/records/7332091",
                    panel_complete=False, measured_panel=None, rna_rows=0,
                    known_blockers=["Real decoded RNA point tables and corresponding gene-code mapping have not been verified",
                                    "PI/cell masks do not establish an independent nuclear mask"],
                    provenance={"source_url": "https://zenodo.org/records/7332091"})
    try:
        sidecars, genuine_candidates, seen = [], [], set()
        with zipfile.ZipFile(input_zip) as archive:
            for entry in archive.infolist():
                path = _safe_name(entry.filename)
                if entry.filename in seen or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Duplicate or symbolic-link ZIP member")
                seen.add(entry.filename)
                if entry.is_dir():
                    continue
                record = {"name": entry.filename, "bytes": entry.file_size, "crc32": f"{entry.CRC:08x}"}
                manifest["members"].append(record)
                if path.name == "._goodPoints.mat":
                    sidecars.append(record)
                elif path.name.lower() in ("goodpoints.mat", "goodreads.mat", "points.csv", "decoded_transcripts.csv"):
                    genuine_candidates.append(record)
        manifest.update(status="blocked_unverified_point_schema" if genuine_candidates else "blocked_missing_molecule_points",
                        appledouble_point_sidecars=sidecars, genuine_point_candidates=genuine_candidates,
                        appledouble_semantics="._goodPoints.mat is a macOS metadata resource fork, not the original MATLAB RNA file")
        _save(destination, manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(destination, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("resolve", "starmap"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--columns-json", help='Explicit names or zero-based indices, e.g. {"x":"x","y":"y","gene":"gene","cell_id":"cell"}')
    parser.add_argument("--header", choices=("present", "absent"), default="present")
    parser.add_argument("--delimiter", choices=("tab", "comma", "space"), default="tab")
    parser.add_argument("--sample-metadata", type=Path)
    parser.add_argument("--geo-soft", type=Path)
    parser.add_argument("--coordinate-unit", default="unverified_source_units")
    parser.add_argument("--coordinate-evidence")
    parser.add_argument("--expected-samples", type=int, default=37)
    parser.add_argument("--rows-per-shard", type=int, default=100000)
    parser.add_argument("--max-member-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--reference-validation-mode", choices=("strict", "report"), default="strict",
                        help="Report records publication-variant differences only in non-training staging; strict rejects them")
    alias_group = parser.add_mutually_exclusive_group()
    alias_group.add_argument("--original-gene-aliases-json", help='Exact allowed original-to-assigned names, e.g. {"FP 48":"FP_48"}')
    alias_group.add_argument("--original-gene-aliases-file", type=Path, help="JSON file containing exact allowed original-to-assigned gene names")
    args = parser.parse_args()
    if args.platform == "starmap":
        result = audit_starmap(args.input, args.output, expected_sha256=args.expected_sha256)
        print(json.dumps(result, indent=2))
        return 4
    if not args.columns_json:
        parser.error("Resolve requires --columns-json after actual RNA header inspection")
    aliases = json.loads(args.original_gene_aliases_file.read_text()) if args.original_gene_aliases_file else json.loads(args.original_gene_aliases_json) if args.original_gene_aliases_json else None
    result = stage_resolve(args.input, args.output, columns=json.loads(args.columns_json),
        delimiter={"tab": "\t", "comma": ",", "space": " "}[args.delimiter], has_header=args.header == "present",
        sample_metadata=args.sample_metadata, geo_soft=args.geo_soft, expected_sha256=args.expected_sha256,
        coordinate_unit=args.coordinate_unit, coordinate_evidence=args.coordinate_evidence,
        expected_samples=args.expected_samples, rows_per_shard=args.rows_per_shard, max_member_bytes=args.max_member_bytes,
        original_gene_aliases=aliases, reference_validation_mode=args.reference_validation_mode)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
