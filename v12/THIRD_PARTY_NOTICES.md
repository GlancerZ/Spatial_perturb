# Third-party sources

This repository does not bundle pretrained scGPT weights or raw experimental datasets.
Downloading a resource does not transfer its license to this repository. Follow each
upstream repository, model provider and dataset provider's terms and attribution.

- scGPT: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT), code commit
  `cebd6fae655b9c585a4807daa3ac31bb764f06b4`, upstream MIT license. The loader reads
  upstream `GeneEncoder` and `ContinuousValueEncoder` definitions; the setup script
  downloads that exact source revision. Preserve upstream copyright/license when
  redistributing downloaded code. Model weights come from the official
  [whole-human model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo)
  and [Google Drive folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y).
  Weight provenance and checksums: `resources/scgpt_human/source_manifest.json`.
- LR source: [Spatialformer ligand–receptor table](https://github.com/TerminatorJ/Spatialformer/blob/f02e7776e8a99c9abdc5fe5572a1a20369faf159/downstream/cell_cell_communication/data/ligand_receptor_database.csv).
  The repository contains the derived single-gene pair table plus its provenance
  manifest in `resources/lr_v6/`; retain those records. The upstream repository
  and constituent biological databases govern their own licenses and citations.
- Raw Xenium and Visium HD data: 10x Genomics and the original study/data providers.
  Source URLs and processing contracts are documented in `docs/DATA.zh.md`.
- Python dependencies retain their respective licenses; see each installed
  distribution's metadata. This notice does not establish a project-code license;
  that remains the repository owner's decision.
