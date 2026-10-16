"""V4 optimizer-boundary, provenance and output-safety regressions (CPU-capable)."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('trainer_v4', ROOT / 'scripts/train_subcellular_v4.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / 'configs/subcellular_geneset_v4_20260919.yaml').read_text())
        self.arch = dict(model_type='subcellular_geneset_v4', slots_per_compartment=8,
                         slot_dim=64, local_pre_layers=1, spatial_layers=2, local_post_layers=2)
        self.sources = {'model.py': 'verified-source'}
        self.state = dict(
            model_version=4, architecture='geneset_feedback_v4', arch_config=copy.deepcopy(self.arch),
            config=copy.deepcopy(self.cfg), config_sha256=trainer.config_hash(self.cfg),
            source_sha256=copy.deepcopy(self.sources), micro_batch_size=16, global_batch_size=64,
            phase='frozen_backbone', step=3, epoch=0, next_update=3, complete=False,
            model={}, optimizer={}, selected_centers=620, backbone_provenance={'weights_sha256': 'official'},
        )

    def validate(self, state=None, cfg=None, **kwargs):
        return trainer.validate_resume(
            self.state if state is None else state, self.cfg if cfg is None else cfg,
            self.sources, self.arch, micro_batch=16, updates_per_epoch=10,
            selected_centers=620, backbone_provenance={'weights_sha256': 'official'}, **kwargs)

    def test_valid_boundary_and_complete_state(self):
        trainer.validate_config(self.cfg)
        self.assertEqual(self.validate(), (3, 0, 3))
        complete = dict(self.state, step=10, epoch=1, next_update=0, complete=True)
        self.assertEqual(self.validate(complete), (10, 1, 0))

    def test_v3_and_unversioned_checkpoints_cannot_resume(self):
        for version, architecture in ((3, 'dual_cls_binned_v3'), (None, 'geneset_feedback_v4')):
            with self.subTest(version=version):
                state = dict(self.state, model_version=version, architecture=architecture)
                with self.assertRaisesRegex(ValueError, 'V3 is incompatible'):
                    self.validate(state)

    def test_slot_and_feedback_architecture_are_bound_to_checkpoint(self):
        for key in ('slots_per_compartment', 'slot_dim', 'local_pre_layers', 'spatial_layers', 'local_post_layers'):
            state = copy.deepcopy(self.state)
            state['arch_config'][key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                self.validate(state)

    def test_loss_and_data_changes_are_rejected_even_with_same_model_shapes(self):
        for section, key, value in (
            ('loss', 'type', 'cross_entropy_plus_bce'),
            ('data', 'paired_mask_ratio', 0.25),
            ('data', 'query_genes', 64),
            ('data', 'center_stride', 16),
            ('protocol', 'bins', 'visible_only'),
            ('training', 'lr_new', 0.0002),
        ):
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'config mismatch'):
                self.validate(cfg=cfg)

    def test_source_assets_and_dataset_identity_guard_resume(self):
        mutations = (
            ('source_sha256', {'model.py': 'different-source'}, 'source mismatch'),
            ('backbone_provenance', {'weights_sha256': 'different'}, 'asset provenance'),
            ('selected_centers', 619, 'dataset size'),
            ('micro_batch_size', 8, 'minibatch size'),
            ('global_batch_size', 32, 'global batch'),
        )
        for key, value, message in mutations:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.validate(dict(self.state, **{key: value}))

    def test_invalid_or_inconsistent_optimizer_cursor_is_rejected(self):
        for changes in (
            dict(step=4), dict(next_update=-1), dict(next_update=10),
            dict(epoch=-1), dict(epoch=2), dict(step=True), dict(complete=True),
            dict(epoch=1, step=11, next_update=1, complete=False),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(dict(self.state, **changes))
        archive = copy.deepcopy(self.state)
        del archive['optimizer']
        with self.assertRaisesRegex(ValueError, 'optimizer state'):
            self.validate(archive)

    def test_frozen_ordinary_backbone_and_single_objective_are_explicit(self):
        for section, key, value in (
            ('loss', 'auxiliary', 0.1), ('loss', 'type', 'bce_mse'),
            ('model', 'backbone_kind', 'scgpt_spatial'),
            ('training', 'backbone_trainable_layers', 1),
        ):
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                trainer.validate_config(cfg)


class RunDirectoryTests(unittest.TestCase):
    def test_default_never_adopts_or_overwrites_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            trainer.prepare_run_directory(run, resume=False)
            existing = run / 'checkpoints' / 'step_00000100.pt'
            existing.write_bytes(b'existing checkpoint')
            with self.assertRaisesRegex(FileExistsError, 'not empty'):
                trainer.prepare_run_directory(run, resume=False)
            self.assertEqual(existing.read_bytes(), b'existing checkpoint')
            with self.assertRaisesRegex(FileNotFoundError, 'resume.pt'):
                trainer.prepare_run_directory(run, resume=True)
            (run / 'resume.pt').write_bytes(b'checkpoint')
            trainer.prepare_run_directory(run, resume=True)
            (run / 'PAUSE_REQUESTED').touch()
            with self.assertRaisesRegex(RuntimeError, 'paused'):
                trainer.prepare_run_directory(run, resume=True)


class OptimizerAndSamplingTests(unittest.TestCase):
    def test_global_windows_and_resume_are_independent_of_physical_gpu_count(self):
        whole = list(trainer.GlobalBatchSampler(137, 64, 16, seed=9))
        for world in (1, 2, 4):
            ranks = [list(trainer.GlobalBatchSampler(137, 64, 16, world, rank, 9)) for rank in range(world)]
            self.assertEqual([batch for micro in zip(*ranks) for batch in micro], whole)
            accumulation = 64 // (16 * world)
            for rank in range(world):
                resumed = trainer.GlobalBatchSampler(137, 64, 16, world, rank, 9)
                resumed.seek(0, 1)
                self.assertEqual(list(resumed), ranks[rank][accumulation:])
        values = sum(whole, [])
        self.assertEqual(len(set(values[:137])), 137)
        self.assertEqual(values[137:], values[:55])

    def test_adamw_dropout_and_learning_rate_resume_match_uninterrupted(self):
        """A real optimizer restart restores moments and does not replay the prefix."""
        torch.manual_seed(123)
        reference = nn.Sequential(nn.Linear(3, 8), nn.GELU(), nn.Dropout(.3), nn.Linear(8, 2))
        interrupted = copy.deepcopy(reference)
        x = torch.arange(57, dtype=torch.float32).reshape(19, 3) / 57
        y = torch.stack((x[:, 0] - x[:, 2], x[:, 1] * 0.2), dim=-1)
        training = dict(warmup_ratio=.3, min_lr_ratio=.1)
        lr = .004

        def run(model, optimizer, start_step, stop_step):
            sampler = trainer.GlobalBatchSampler(19, 8, 2, seed=7)
            step = start_step
            consumed = []
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(start_step // sampler.updates, 2):
                epoch_start = start_step % sampler.updates if epoch == start_step // sampler.updates else 0
                sampler.seek(epoch, epoch_start)
                for offset, indices in enumerate(sampler):
                    update = epoch_start + offset // sampler.accumulation
                    micro = offset % sampler.accumulation
                    torch.manual_seed(trainer.logical_forward_seed(7, epoch, update, micro))
                    loss = (model(x[indices]) - y[indices]).square().mean()
                    (loss / sampler.accumulation).backward()
                    consumed.extend(indices)
                    if micro + 1 == sampler.accumulation:
                        for group in optimizer.param_groups:
                            group['lr'] = lr * trainer.lr_multiplier(step + 1, 6, training)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        if step == stop_step:
                            return consumed
            return consumed

        optimizer_ref = torch.optim.AdamW(reference.parameters(), lr=lr)
        all_indices = run(reference, optimizer_ref, 0, 6)
        optimizer_first = torch.optim.AdamW(interrupted.parameters(), lr=lr)
        first_indices = run(interrupted, optimizer_first, 0, 4)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'resume.pt'
            trainer.atomic_save(dict(model=interrupted.state_dict(), optimizer=optimizer_first.state_dict()), checkpoint)
            state = torch.load(checkpoint, weights_only=False)
        restored = copy.deepcopy(interrupted)
        restored.load_state_dict(state['model'])
        optimizer_second = torch.optim.AdamW(restored.parameters(), lr=lr)
        optimizer_second.load_state_dict(state['optimizer'])
        second_indices = run(restored, optimizer_second, 4, 6)
        self.assertEqual(first_indices + second_indices, all_indices)
        for expected, actual in zip(reference.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, expected in optimizer_ref.state_dict()['state'].items():
            for name, value in expected.items():
                torch.testing.assert_close(optimizer_second.state_dict()['state'][key][name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
