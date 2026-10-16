"""Small fake proofs for relocation boundaries; no torch/assets/checkpoint needed.

Run only where project execution is authorized. These tests validate the audit
helpers, not a real checkpoint load or inference result.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from load_v12_portable import (
    fingerprint, require_compute_execution, runtime_config,
    verify_asset_bytes, verify_portable_provenance, verify_sources,
)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


class RelocationProofs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / 'assets'
        self.files = {
            'code/scgpt/model/model.py': b'# fake upstream source\n',
            'scGPT_human/best_model.pt': b'fake checkpoint bytes, not torch\n',
            'scGPT_human/args.json': b'{"fake": true}\n',
            'scGPT_human/vocab.json': b'{"L": 0, "R": 1}\n',
            'source_manifest.json': b'{"origin": "fake-proof"}\n',
        }
        for name, payload in self.files.items():
            path = self.assets / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        self.provenance = {
            'backbone_kind': 'scgpt', 'source_repository': 'https://github.com/bowang-lab/scGPT',
            'source': '/original/assets/code/scgpt/model/model.py',
            'weights': '/original/assets/scGPT_human/best_model.pt',
            'source_sha256': sha(self.files['code/scgpt/model/model.py']),
            'weights_sha256': sha(self.files['scGPT_human/best_model.pt']),
            'weights_md5': 'fake-md5-for-equality-proof',
            'args_sha256': sha(self.files['scGPT_human/args.json']),
            'vocab_sha256': sha(self.files['scGPT_human/vocab.json']),
            'loaded_encoder_tensors': 7,
            'encoder_architecture': {'layers': 2, 'norm': 'post'},
            'source_manifest': {
                'path': '/original/assets/source_manifest.json',
                'sha256': sha(self.files['source_manifest.json']),
                'backbone_kind': 'scgpt', 'model_name': 'whole-human',
                'repository': 'https://github.com/bowang-lab/scGPT',
                'code_commit': 'fake-origin',
                'weights_sha256': sha(self.files['scGPT_human/best_model.pt']),
            },
        }
        self.lr = self.root / 'curated.csv'
        self.lr.write_bytes(b'ligand,receptor,weight\nL,R,1\n')
        self.config = {'assets_dir': '/original/assets', 'model': {'backbone_kind': 'scgpt'},
                       'lr': {'enabled': True, 'path': '/original/curated.csv',
                              'sha256': sha(self.lr.read_bytes()), 'source': 'fake-origin'}}

    def test_identical_bytes_work_from_a_second_directory(self):
        relocated = self.root / 'different/server/assets'
        shutil.copytree(self.assets, relocated)
        verify_asset_bytes(self.provenance, relocated)
        actual = deepcopy(self.provenance)
        actual['source'] = str(relocated / 'code/scgpt/model/model.py')
        actual['weights'] = str(relocated / 'scGPT_human/best_model.pt')
        actual['source_manifest']['path'] = str(relocated / 'source_manifest.json')
        before = deepcopy(self.provenance)
        paths = verify_portable_provenance(self.provenance, actual)
        self.assertEqual(set(paths), {'source', 'weights', 'source_manifest.path'})
        self.assertEqual(self.provenance, before)

    def test_every_asset_byte_change_is_rejected(self):
        for name, original in self.files.items():
            with self.subTest(asset=name):
                path = self.assets / name
                path.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    verify_asset_bytes(self.provenance, self.assets)
                path.write_bytes(original)

    def test_non_path_provenance_changes_are_rejected(self):
        mutations = [('weights_md5', 'different'), ('loaded_encoder_tensors', 8),
                     ('source_sha256', '1' * 64), ('extra', 'unapproved')]
        for key, value in mutations:
            with self.subTest(field=key):
                actual = deepcopy(self.provenance)
                actual[key] = value
                with self.assertRaises(ValueError):
                    verify_portable_provenance(self.provenance, actual)
        for key, value in [('sha256', '1' * 64), ('code_commit', 'different'),
                           ('model_name', 'spatial'), ('extra', 'unapproved')]:
            with self.subTest(manifest=key):
                actual = deepcopy(self.provenance)
                actual['source_manifest'][key] = value
                with self.assertRaises(ValueError):
                    verify_portable_provenance(self.provenance, actual)
        actual = deepcopy(self.provenance)
        actual['encoder_architecture']['layers'] = 3
        with self.assertRaises(ValueError):
            verify_portable_provenance(self.provenance, actual)

    def test_config_relocation_is_nonmutating_and_preserves_identity(self):
        original = deepcopy(self.config)
        expected_sha = fingerprint(original)
        cfg, paths = runtime_config(self.config, expected_sha, self.assets, self.lr)
        self.assertEqual(self.config, original)
        self.assertEqual(cfg['lr']['sha256'], original['lr']['sha256'])
        self.assertEqual(cfg['lr']['source'], original['lr']['source'])
        self.assertEqual(cfg['model'], original['model'])
        self.assertEqual(set(paths), {'assets_dir', 'lr.path'})
        self.assertNotEqual(fingerprint(cfg), expected_sha)
        tampered = deepcopy(self.config)
        tampered['model']['dropout'] = 0.5
        with self.assertRaises(ValueError):
            runtime_config(tampered, expected_sha, self.assets, self.lr)
        self.lr.write_bytes(self.lr.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            runtime_config(self.config, expected_sha, self.assets, self.lr)

    def test_all_source_entries_and_root_boundary_are_enforced(self):
        files = {'src/spatial_pretrain/models/subcellular_v12.py': b'model\n',
                 'src/spatial_pretrain/data/subcellular_v12.py': b'data\n',
                 'scripts/also_required.py': b'helper\n'}
        sources = {}
        for name, payload in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            sources[name] = sha(payload)
        verify_sources(sources, self.root)
        (self.root / 'scripts/also_required.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify_sources(sources, self.root)
        (self.root / 'scripts/also_required.py').write_bytes(files['scripts/also_required.py'])
        sources['../outside.py'] = sha(b'outside')
        with self.assertRaises(ValueError):
            verify_sources(sources, self.root)

    def test_standalone_audit_has_no_fabricated_slurm_id(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch('socket.gethostname', return_value='cuda-workstation'), \
                patch('socket.getfqdn', return_value='cuda-workstation.example'), \
                patch('pathlib.Path.exists', return_value=False):
            audit = require_compute_execution(standalone_compute=True)
            self.assertIsNone(audit['slurm_job_id'])
            self.assertIsNone(audit['slurm_step_id'])
            self.assertEqual(audit['hostname'], 'cuda-workstation')
            with self.assertRaises(RuntimeError):
                require_compute_execution()
        with patch.dict('os.environ', {}, clear=True), \
                patch('socket.gethostname', return_value='rorqual-login01'), \
                patch('socket.getfqdn', return_value='rorqual-login01.example'):
            with self.assertRaises(RuntimeError):
                require_compute_execution(standalone_compute=True)


if __name__ == '__main__':
    unittest.main()
