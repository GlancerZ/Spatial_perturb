#!/usr/bin/env bash
# Choose a single dataset explicitly; never bulk-download the entire catalog.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: bash scripts/download_data.sh MANIFEST.tsv DESTINATION SAMPLE_OR_DATASET' >&2
  exit 2
fi
manifest=$1
destination=$2
sample=$3
[[ -f "$manifest" && -n "$sample" ]] || exit 2
rows=$(awk -F '\t' -v sample="$sample" '
  NR==1 {for(i=1;i<=NF;i++)c[$i]=i; if(!c["sample_or_dataset"]||!c["download_url"])exit 2; next}
  $(c["sample_or_dataset"])==sample && $(c["download_url"])!="" {print $(c["download_url"])}
' "$manifest")
[[ -n "$rows" ]] || { echo 'No direct files for that dataset; consult its source page and access notes.' >&2; exit 2; }
safe_sample=$(printf '%s' "$sample" | tr -c 'A-Za-z0-9_.-' '_')
while IFS= read -r url; do
  [[ "$url" == https://* || "$url" == http://* || "$url" == ftp://* ]] || exit 2
  clean=${url%%\?*}
  if [[ "$clean" == *10xgenomics.com/samples/* || "$clean" == *amazonaws.com/10x.files/samples/* ]]; then
    relative="samples/${clean#*/samples/}"
  else
    relative="$safe_sample/${clean##*/}"
  fi
  [[ "$relative" != *'/../'* && "$relative" != */ && "$relative" != /* ]] || exit 2
  output="$destination/$relative"
  if [[ -e "$output" ]]; then
    echo "EXISTS (not verified by this script): $output"
    continue
  fi
  mkdir -p -- "$(dirname -- "$output")"
  curl --fail --location --retry 2 --connect-timeout 30 "$url" --output "$output.partial"
  [[ -s "$output.partial" ]] || { echo 'Empty download' >&2; exit 1; }
  mv --no-clobber -- "$output.partial" "$output"
  sha256sum "$output"
done <<< "$rows"
