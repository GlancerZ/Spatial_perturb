"""Strict CPU loaders with explicit ordinary-scGPT versus historical-spatial assets.

Only upstream GeneEncoder / ContinuousValueEncoder definitions are executed;
legacy FlashAttention, torchtext and unused reconstruction heads are unnecessary.
The standard PyTorch post-norm Transformer path is the upstream fallback path.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


BACKBONE_LAYOUTS = {
    'scgpt': ('scGPT_human', 'scgpt', 'https://github.com/bowang-lab/scGPT'),
    'scgpt_spatial': ('scGPT_spatial_v1', 'scgpt_spatial', 'https://github.com/bowang-lab/scGPT-spatial'),
}
KNOWN_SPATIAL_V1_SHA256 = '1ce6a9eaa7cc199b23d404ea38203956d04699aaede42b6ea67aa0bf3a88c8be'


def verify_source_manifest(asset_dir, backbone_kind, weights_sha256, vocab_sha256, args_sha256):
    """Validate declared download provenance against bytes, never tensor names.

    Ordinary whole-human scGPT also contains flag_encoder weights, so that key
    cannot distinguish it from scGPT-spatial. The downloaded-file hash can.
    """
    if backbone_kind != 'scgpt':
        return None
    if weights_sha256 == KNOWN_SPATIAL_V1_SHA256:
        raise ValueError('Ordinary scGPT cannot load the known scGPT-spatial V1 checkpoint hash')
    path = Path(asset_dir) / 'source_manifest.json'
    if not path.is_file():
        raise FileNotFoundError('Ordinary scGPT requires source_manifest.json with verified whole-human download provenance')
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or manifest.get('backbone_kind') != 'scgpt'
            or manifest.get('model_name') != 'whole-human'
            or manifest.get('repository') != BACKBONE_LAYOUTS['scgpt'][2]):
        raise ValueError('Source manifest does not identify official ordinary scGPT whole-human')
    if manifest.get('weights_sha256') != weights_sha256:
        raise ValueError('Checkpoint SHA256 does not match the ordinary scGPT source manifest')
    for key, actual in [('vocab_sha256', vocab_sha256), ('args_sha256', args_sha256)]:
        if key in manifest and manifest[key] != actual:
            raise ValueError(f'{key} does not match the ordinary scGPT source manifest')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                backbone_kind=manifest['backbone_kind'], model_name=manifest['model_name'],
                repository=manifest['repository'], code_commit=manifest.get('code_commit'),
                weights_sha256=manifest['weights_sha256'])


def resolve_backbone_assets(asset_dir: str | Path, backbone_kind: str):
    """Choose exactly one declared source; never search/fallback to another model."""
    if backbone_kind not in BACKBONE_LAYOUTS:
        raise ValueError(f'Unknown backbone_kind={backbone_kind!r}; choose an explicit supported kind')
    directory, package, repository = BACKBONE_LAYOUTS[backbone_kind]
    root = Path(asset_dir)
    weights = root / directory
    source = root / 'code' / package / 'model/model.py'
    required = [source, *(weights / name for name in ('args.json', 'vocab.json', 'best_model.pt'))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{backbone_kind} requires its own assets (no fallback): {missing}')
    return weights, source, repository


class ScGPTBackbone(nn.Module):
    def __init__(self, asset_dir: str | Path, checkpointing: bool = True,
                 backbone_kind: str = 'scgpt_spatial'):
        super().__init__()
        # The default preserves existing saved spatial configurations. Every new
        # ordinary-scGPT experiment explicitly supplies backbone_kind='scgpt'.
        self.backbone_kind = backbone_kind
        weights, source, repository = resolve_backbone_assets(asset_dir, backbone_kind)
        self.config = json.loads((weights / 'args.json').read_text())
        self.vocab = json.loads((weights / 'vocab.json').read_text())
        cfg = self.config
        if cfg.get('input_emb_style', 'continuous') != 'continuous':
            raise ValueError('V2 requires the official continuous value encoder')
        if cfg.get('pre_norm', False):
            raise ValueError('This verified standard encoder path requires a post-norm checkpoint')
        if cfg.get('domain_spec_batchnorm', False) or cfg.get('DSBN', False):
            raise ValueError('A checkpoint with domain-specific normalization needs an explicit supported encoder')
        if cfg['pad_token'] not in self.vocab or '<cls>' not in self.vocab:
            raise ValueError('Official vocabulary lacks a required special token')
        if set(self.vocab.values()) != set(range(len(self.vocab))):
            raise ValueError('Official vocabulary IDs must be contiguous; do not silently alter pretrained rows')
        namespace = {'nn': nn, 'Tensor': Tensor, 'torch': torch, 'Optional': Optional}
        parsed = ast.parse(source.read_text())
        definitions = [node for node in parsed.body if isinstance(node, ast.ClassDef)
                       and node.name in ('GeneEncoder', 'ContinuousValueEncoder')]
        if len(definitions) != 2:
            raise ValueError('Cannot identify the two official embedding definitions')
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
        self.dim = int(cfg['embsize'])
        self.nheads = int(cfg['nheads'])
        self.pad_id = int(self.vocab[cfg['pad_token']])
        self.encoder = namespace['GeneEncoder'](len(self.vocab), self.dim, padding_idx=self.pad_id)
        self.value_encoder = namespace['ContinuousValueEncoder'](self.dim, cfg['dropout'])
        layer = nn.TransformerEncoderLayer(self.dim, self.nheads, cfg['d_hid'],
                                           cfg['dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, cfg['nlayers'], enable_nested_tensor=False)
        # Hash the exact selected checkpoint. Ordinary scGPT never reads a
        # scGPT-spatial checkpoint or searches an alternate weights directory.
        weights_path = weights / 'best_model.pt'
        digest = hashlib.sha256()
        md5 = hashlib.md5()
        with weights_path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
                md5.update(block)
        args_sha256 = hashlib.sha256((weights / 'args.json').read_bytes()).hexdigest()
        vocab_sha256 = hashlib.sha256((weights / 'vocab.json').read_bytes()).hexdigest()
        source_manifest = verify_source_manifest(asset_dir, backbone_kind, digest.hexdigest(),
                                                  vocab_sha256, args_sha256)
        raw = torch.load(weights_path, map_location='cpu', weights_only=False)
        if any(key.startswith(('dsbn.', 'bn.')) for key in raw):
            raise ValueError('Checkpoint contains an additional normalization encoder that cannot be discarded')
        prefixes = ('encoder.', 'value_encoder.', 'transformer_encoder.')
        state = {key.replace('Wqkv.', 'in_proj_'): value for key, value in raw.items()
                 if key.startswith(prefixes)}
        expected = self.state_dict()
        missing = [key for key, value in expected.items()
                   if key not in state or state[key].shape != value.shape]
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise ValueError(f'Official encoder incompatibility: missing/mismatched={missing}; extra={unexpected}')
        self.load_state_dict(state, strict=True)
        self.checkpointing = bool(checkpointing)
        self.provenance = {
            'backbone_kind': backbone_kind, 'source_repository': repository,
            'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'weights': str(weights_path), 'weights_sha256': digest.hexdigest(),
            'weights_md5': md5.hexdigest(), 'loaded_encoder_tensors': len(state),
            'args_sha256': args_sha256, 'vocab_sha256': vocab_sha256,
            'source_manifest': source_manifest,
            'encoder_architecture': dict(dim=self.dim, heads=self.nheads, layers=int(cfg['nlayers']),
                                         feedforward_dim=int(cfg['d_hid']), value_encoder='continuous',
                                         norm='post', vocabulary_size=len(self.vocab)),
            'encoder_load': 'strict; Wqkv to in_proj; all other encoder names unchanged',
        }

    def forward(self, genes: Tensor, values: Tensor, padding_mask: Tensor, compartment: Tensor) -> Tensor:
        hidden = self.encoder(genes) + self.value_encoder(values) + compartment
        for layer in self.transformer_encoder.layers:
            # Frozen parameters must not detach the trainable compartment input.
            if self.checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, src_key_padding_mask=padding_mask, use_reentrant=False)
            else:
                hidden = layer(hidden, src_key_padding_mask=padding_mask)
        if self.transformer_encoder.norm is not None:
            hidden = self.transformer_encoder.norm(hidden)
        return hidden
