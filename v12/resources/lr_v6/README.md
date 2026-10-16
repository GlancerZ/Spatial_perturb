# V6 shared directed LR prior

The preparation command (compute node only) creates an immutable normalized CSV,
manifest and both experiment configurations. Nothing has been converted or
validated by the presence of this README alone.

Source: the CellChat-shaped curated table vendored in
[Spatialformer revision f02e777](https://github.com/TerminatorJ/Spatialformer/blob/f02e7776e8a99c9abdc5fe5572a1a20369faf159/downstream/cell_cell_communication/data/ligand_receptor_database.csv).
Original table SHA256: `791002fdbc810676306c3e89b883895da01620651c015931423d627d525d1b43`.
Retain source attribution and upstream license; no model weights are reused.

Preparation keeps only single-gene ligand/receptor symbols and curated secreted,
cell-contact or ECM annotations. It excludes complexes as whole rows and never
expands an AND-complex into independent subunit interactions. The resulting
subset is deliberately incomplete. Duplicate directed pairs get weight one,
regardless of their number of publications or pathway annotations. No predicted
PPI rows are added. This is not a full reproduction of CellNEST's database.

Runtime matches genes to ordinary scGPT's vocabulary and reports skipped genes.
Visible cytoplasmic counts, normalized using only visible panel counts, determine
sender-ligand/receiver-receptor activity. RNA is an expression proxy, not measured
secreted ligand or surface receptor protein.

`lr_bias[i,j] = sum(weight * activity_ligand_j * activity_receptor_i)
/ (known_pair_weight[i,j] + 4)`; terms are included only when both endpoints are
measured and unmasked. Thus finite evidence is shrunk and absent evidence gives
zero. `lr_coverage` separately reports known pair weight divided by total prior
weight; the model must not multiply it into the bias again. Real measured zeros
contribute to the known denominator, while unknown/masked genes do not.
Evidence uses the full visible panel even when gene tokens are length-truncated;
this extra observed input is shared by A and B and recorded explicitly.
