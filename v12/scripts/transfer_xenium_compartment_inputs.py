"""Copy only existing compressed ZIP-member bytes and references to Vulcan.

No transcript decoding or preprocessing is done here. The compute-node pipeline
opens and validates the resulting single-member ZIPs.
"""
from copy import copy
import csv
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "vulcan:/scratch/glancerz/spatial-pretrain-v3/"
INPUTS = Path("data/pretrain_data/10x_xenium/compartments/inputs")


def copy_compressed_member(source, member, destination):
    with zipfile.ZipFile(source) as original:
        info = original.getinfo(member)
        next_offsets = [x.header_offset for x in original.infolist() if x.header_offset > info.header_offset]
        end = min(next_offsets, default=original.start_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_suffix(".zip.partial")
        with open(source, "rb") as src, zipfile.ZipFile(staged, "w", allowZip64=True) as dst:
            src.seek(info.header_offset)
            remaining = end - info.header_offset
            while remaining:
                block = src.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise EOFError(source)
                dst.fp.write(block)
                remaining -= len(block)
            # Preserve compression, CRC, size and local header; relocate its
            # central-directory entry. No decompression/recompression is needed.
            relocated = copy(info)
            relocated.header_offset = 0
            dst.filelist.append(relocated)
            dst.NameToInfo[relocated.filename] = relocated
            dst.start_dir = dst.fp.tell()
            dst._didModify = True
        staged.replace(destination)


def main():
    rows = list(csv.DictReader((ROOT / "data/pretrain_data/manifests/xenium49_compartment_inputs.tsv").open(), delimiter="\t"))
    transfers = []
    for row in rows:
        with zipfile.ZipFile(ROOT / row["archive"]) as z:
            choices = [i for i in z.infolist() if Path(i.filename).name in ("transcripts.parquet", "transcripts.csv.gz")]
            selected = min(choices, key=lambda i: i.compress_size)
        transfers.append((selected.compress_size, row.copy(), selected.filename))
    manifest = ROOT / "data/pretrain_data/manifests/xenium49_compartment_vulcan_inputs.tsv"
    with manifest.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        for _, row, member in transfers:
            row = row.copy()
            row["archive"] = str(INPUTS / (row["sample"] + ".zip"))
            row["transcript_files"] = member
            writer.writerow(row)
    subprocess.run(["rsync", "-aL", "--relative", str(manifest.relative_to(ROOT)), REMOTE], cwd=ROOT, check=True)
    for idx, (size, row, member) in enumerate(sorted(transfers, key=lambda x: x[0]), 1):
        relative = INPUTS / (row["sample"] + ".zip")
        target = ROOT / relative
        ready = ROOT / INPUTS / ".ready" / (row["sample"] + ".done")
        print(f"[transfer] {idx}/49 {row['sample']} compressed_GiB={size/2**30:.3f}", flush=True)
        if not target.exists():
            copy_compressed_member(ROOT / row["archive"], member, target)
        subprocess.run(["rsync", "-aL", "--partial", "--relative", str(relative), row["existing_h5ad"], REMOTE], cwd=ROOT, check=True)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(f"{size}\n")
        subprocess.run(["rsync", "-a", "--relative", str(ready.relative_to(ROOT)), REMOTE], cwd=ROOT, check=True)
        print(f"[transferred] {row['sample']}", flush=True)
    print("[transfer-complete] 49/49", flush=True)


if __name__ == "__main__":
    main()
