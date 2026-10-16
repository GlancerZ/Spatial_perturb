"""Source selection must be explicit and a partial pretrained encoder must fail."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spatial_pretrain.models.scgpt_backbone import (ScGPTBackbone, resolve_backbone_assets,
    verify_source_manifest, KNOWN_SPATIAL_V1_SHA256)


SOURCE = '''
class GeneEncoder(nn.Module):
    def __init__(self, count, dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(count, dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(dim)
    def forward(self, genes):
        return self.enc_norm(self.embedding(genes))

class ContinuousValueEncoder(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, values):
        return self.norm(self.linear2(self.linear1(values.unsqueeze(-1)).relu()))
'''


class BackboneSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def assets(self, kind):
        folder, package = ('scGPT_human', 'scgpt') if kind == 'scgpt' else ('scGPT_spatial_v1', 'scgpt_spatial')
        weights = self.root / folder
        weights.mkdir()
        source = self.root / 'code' / package / 'model/model.py'
        source.parent.mkdir(parents=True)
        source.write_text(SOURCE)
        (weights / 'args.json').write_text(json.dumps(dict(embsize=16, nheads=4, d_hid=32,
            nlayers=2, pad_token='<pad>', pad_value=-2, dropout=0.0, input_emb_style='continuous')))
        (weights / 'vocab.json').write_text(json.dumps({'<pad>': 0, '<cls>': 1, 'A': 2, 'B': 3}))
        (weights / 'best_model.pt').write_bytes(b'unit-test-only; torch.load is mocked')
        if kind == 'scgpt':
            (self.root / 'source_manifest.json').write_text(json.dumps(dict(
                backbone_kind='scgpt', model_name='whole-human', repository='https://github.com/bowang-lab/scGPT',
                weights_sha256=hashlib.sha256((weights / 'best_model.pt').read_bytes()).hexdigest(),
                vocab_sha256=hashlib.sha256((weights / 'vocab.json').read_bytes()).hexdigest())))
        return weights, source

    def test_ordinary_choice_never_falls_back_to_available_spatial_assets(self):
        self.assets('scgpt_spatial')
        with self.assertRaisesRegex(FileNotFoundError, 'no fallback'):
            resolve_backbone_assets(self.root, 'scgpt')

    def test_each_kind_selects_its_own_source_even_when_both_exist(self):
        ordinary = self.assets('scgpt')
        spatial = self.assets('scgpt_spatial')
        for kind, expected in [('scgpt', ordinary), ('scgpt_spatial', spatial)]:
            selected_weights, selected_source, _ = resolve_backbone_assets(self.root, kind)
            self.assertEqual((selected_weights, selected_source), expected)
        with self.assertRaisesRegex(ValueError, 'Unknown backbone_kind'):
            resolve_backbone_assets(self.root, 'auto')

    def test_missing_encoder_tensor_cannot_silently_use_random_initialization(self):
        self.assets('scgpt')
        with patch('spatial_pretrain.models.scgpt_backbone.torch.load', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Official encoder incompatibility'):
                ScGPTBackbone(self.root, backbone_kind='scgpt')

    def test_renamed_spatial_checkpoint_is_rejected_by_ordinary_loader(self):
        weights, _ = self.assets('scgpt')
        # A different file copied under the ordinary name does not acquire the
        # verified ordinary model's provenance simply by sharing tensor names.
        (weights / 'best_model.pt').write_bytes(b'another-model-copied-under-the-ordinary-name')
        with patch('spatial_pretrain.models.scgpt_backbone.torch.load') as load:
            with self.assertRaisesRegex(ValueError, 'SHA256 does not match'):
                ScGPTBackbone(self.root, backbone_kind='scgpt')
        load.assert_not_called()

    def test_manifest_identity_and_known_spatial_hash_are_enforced(self):
        weights, _ = self.assets('scgpt')
        digest = hashlib.sha256((weights / 'best_model.pt').read_bytes()).hexdigest()
        vocab_digest = hashlib.sha256((weights / 'vocab.json').read_bytes()).hexdigest()
        verified = verify_source_manifest(self.root, 'scgpt', digest, vocab_digest, 'unused')
        self.assertEqual(verified['model_name'], 'whole-human')
        with self.assertRaisesRegex(ValueError, 'known scGPT-spatial'):
            verify_source_manifest(self.root, 'scgpt', KNOWN_SPATIAL_V1_SHA256, vocab_digest, 'unused')
        manifest_path = self.root / 'source_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['backbone_kind'] = 'scgpt_spatial'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'does not identify'):
            verify_source_manifest(self.root, 'scgpt', digest, vocab_digest, 'unused')

    def test_missing_manifest_cannot_silently_skip_provenance(self):
        self.assets('scgpt')
        (self.root / 'source_manifest.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'requires source_manifest'):
            verify_source_manifest(self.root, 'scgpt', 'not-a-spatial-hash', 'unused', 'unused')


if __name__ == '__main__':
    unittest.main()
