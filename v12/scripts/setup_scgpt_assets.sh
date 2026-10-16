#!/usr/bin/env bash
# Download ordinary whole-human scGPT assets into the strict loader layout.
set -euo pipefail
asset_dir="${1:-resources/scgpt_official}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$asset_dir" ]]; then
  echo 'Destination already exists; choose a new directory. Existing assets are never overwritten.' >&2
  exit 2
fi
command -v curl >/dev/null
command -v gdown >/dev/null
mkdir -p "$asset_dir/scGPT_human" "$asset_dir/code/scgpt/model"
commit=cebd6fae655b9c585a4807daa3ac31bb764f06b4
curl -fL --retry 2 "https://raw.githubusercontent.com/bowang-lab/scGPT/$commit/scgpt/model/model.py" -o "$asset_dir/code/scgpt/model/model.py"
curl -fL --retry 2 "https://raw.githubusercontent.com/bowang-lab/scGPT/$commit/LICENSE" -o "$asset_dir/code/LICENSE.scGPT"
gdown 1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1 -O "$asset_dir/scGPT_human/args.json"
gdown 14AebJfGOUF047Eg40hk57HCtrb0fyDTm -O "$asset_dir/scGPT_human/best_model.pt"
gdown 1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC -O "$asset_dir/scGPT_human/vocab.json"
(
 cd -- "$asset_dir/scGPT_human"
 sha256sum -c <<'SHA'
c18e075e018140cb8b2d9029387b9de26607a5ce6a8ccabd6ead70cd76b95d60  args.json
6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922  best_model.pt
acca93d114ca62c3f0f50debbd23e8c87f0714f4737764454f6b2b13f2e8580f  vocab.json
SHA
)
# Preserve the historical provenance schema expected by verify_source_manifest.
cp -- "$repo_root/resources/scgpt_human/source_manifest.json" "$asset_dir/source_manifest.json"
echo "Verified ordinary scGPT assets: $asset_dir"
