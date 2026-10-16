> **Current model: V12 (2026-10-16)** — [code and overview](v12/README.md), [installation and usage](v12/docs/INSTALL.zh.md), [all recorded data download sources](v12/docs/DATA.zh.md), [model details](v12/docs/MODEL.zh.md). The earlier model below is retained for reference.

# Spatial Perturb

Spatial transcriptomics pretraining with nuclear and cytoplasmic gene tokens, masked expression reconstruction, and cytoplasmic cell-to-cell attention.

## Features

- Variable expressed-gene tokens and masked mean pooling that excludes padding.
- Optional relative spatial attention bias applied only during cytoplasmic inter-cell attention.
- Separate learnable angle embeddings (16 sectors) and distance embeddings (7 categories by default).
- Shared expression-bin boundaries fitted on training cells and saved in the configuration/checkpoint for consistent evaluation.
- AdamW/Muon training, resumable checkpoints, and frozen-embedding evaluation.

## Setup

```bash
pip install -e .
```

The example configuration uses placeholder data paths. Supply compartment h5ad data and build caches using the training pipeline before fitting shared bins. Gene initialization is optional and disabled in the portable example.

## Spatial model

For receiver i and sender j, compute coords[j] - coords[i]. The angle covers [0, 2π) in 16 sectors. Distance boundaries default to 10, 20, 40, 80, 160 and 320 micrometres; set `cell_coordinate_scale` to convert your coordinate units to micrometres. The two embedding tables add per-head biases to attention logits, not to initial gene tokens. Coincident cells receive no angular bias. Spatial embeddings start at zero. Translation invariance is preserved; rotation invariance is not imposed.

Create a new run configuration from existing compartment caches:

```bash
python scripts/prepare_spatial_mean_config.py --base-config configs/xenium_compartment_smoke.yaml --output-config configs/spatial.yaml --output-dir outputs/spatial --coordinate-scale 1
python scripts/train_pretrain.py --config configs/spatial.yaml
```

The preparation script fits boundaries from sampled positive training expression only, enables spatial attention and leaves existing cache arrays unchanged. These boundaries are embedded in the saved configuration. Legacy checkpoints without shared boundaries retain a warned approximation during evaluation; retraining is required for the new binning contract.

Spatially biased cell attention uses automatically dispatched SDPA. Within-cell token attention retains its existing Flash setting. Measure GPU memory and throughput before selecting a production batch size.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_spatial_bias.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_compartment*.py'
```

The new spatial/binned-expression changes have passed Python syntax checks. Tensor tests and GPU validation have not yet been executed for these changes. This upload contains source code, tests, portable configuration and entry points; no datasets, trained weights, cluster launch settings are included. Experiment notes are provided separately below.

## Experiment notes

- [Shared expression binning and relative spatial encoding (Chinese)](docs/cytoplasm_spatial_binning_v2.zh.md)
- [Masked-mean model run history (Chinese; historical snapshot)](docs/cytoplasm_mean_live_run.zh.md)
- [Completed Muon checkpoint validation (Chinese)](docs/cytoplasm_mean_muon_checkpoint_validation.zh.md)

The checkpoint report evaluates the earlier model on a fixed validation subset. It is not a result for the new spatial-encoding model or a full independent test benchmark. Historical run notes include environment-specific paths and job records for reproducibility; these are not portable setup instructions.
