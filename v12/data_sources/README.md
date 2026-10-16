# Public data source catalog

This directory contains source links and metadata, not raw datasets. See [中文数据说明](../docs/DATA.zh.md).

| File | Scope |
|---|---|
| `v12_xenium48.tsv` | Exact 48 samples named by the V12 base config; one official output archive per sample |
| `visium_hd5.tsv` | Five planned mixed-continuation samples, two required files each |
| `benchmark_sources.tsv` | GSE316387, LungPF, CRC expression/annotation records, HiVis liver M6 |
| `collected_sources.tsv` | Broader recorded collection and candidates; these are not all V12 training inputs |

All TSVs use `platform`, `usage`, `sample_or_dataset`, `source_page`, `download_url`, `notes`. Empty `download_url` means use the linked record or resolve the documented access/input limitation. A catalog row is a file/source reference, not an independent sample or patient. Repeated URLs across purposes are intentional; sources may also overlap biologically.

`V12_training` applies only to `v12_xenium48.tsv`. `planned_mixed_continuation` records the documented Visium HD plan. Other usage values describe broader collection, missing inputs, or benchmark scope. Historical HTTP results and checksum statements describe the recorded acquisition, not fresh checks on publication day. Links have not all been rechecked online.

The two V12 samples `Xenium_V1_hBoneMarrow_nondiseased_section` and `Xenium_V1_hBone_nondiseased_section` have URLs reconstructed from the recorded provider-relative archive paths and the project's official 10x CDN pattern; their exact dataset landing pages were not retained. Other V12 links come from recorded 10x page/file inventories, including the supplementary missing-source inventory. See the Chinese guide for provenance and preprocessing requirements.

Download only selected inputs to your local data root. For example, extract the HD5 URLs without executing any transfer:

```bash
awk -F '\t' 'NR > 1 && $5 != "" { print $5 }' data_sources/visium_hd5.tsv
```

Do not blindly download every collected row: it includes images, alternate archives, candidate assets, duplicates, model-source inventories and annotation references. Respect the original provider's license and access requirements; this repository does not redistribute their data or grant a replacement license.

`visium_hd5.sha256` preserves the ten historical acquisition/transfer content hashes with provider-relative paths. Run `sha256sum -c data_sources/visium_hd5.sha256` from a download root containing `samples/spatial-exp/...`; these are recorded local content hashes, not a claim that the publisher issued SHA256 values.
