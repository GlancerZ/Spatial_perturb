"""Run the strict marker-gated proximity audit across available lung H5ADs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnose_xenium_spillover import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--epcam-max-target", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(args.h5ad_dir.glob("*Lung*.h5ad")) + sorted(args.h5ad_dir.glob("*lung*.h5ad")):
        if any(item["sample"] == path.stem for item in records):
            continue
        suffix = "strict_audit" if args.epcam_max_target is None else f"strict_epcam{args.epcam_max_target}_audit"
        output = args.output_dir / f"{path.stem}_{suffix}.json"
        try:
            result = (json.loads(output.read_text()) if output.exists() else
                      run(path, strict_gate=True, epcam_max_target=args.epcam_max_target))
            if not output.exists():
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            held_out = [gene for gene in ("CEACAM6", "MUC1") if gene in result["readouts"]]
            records.append({"sample": path.stem, "status": "ok", "held_out": held_out,
                            "near": result["near_far"]["near_le15um"],
                            "far": result["near_far"]["far_gt50um"],
                            "ratios": {gene: result["readouts"][gene]["depth_area_standardization"].get(
                                "depth_area_standardized_risk_ratio") for gene in held_out}})
        except (ValueError, KeyError, OSError) as exc:
            records.append({"sample": path.stem, "status": "not_evaluable", "reason": str(exc)})
    summary = args.output_dir / f"lung_panel_{suffix}_summary.json"
    if summary.exists():
        raise FileExistsError(summary)
    summary.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    print(json.dumps(records, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
