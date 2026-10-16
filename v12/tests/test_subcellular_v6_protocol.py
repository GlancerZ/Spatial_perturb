"""V6 experiment, checkpoint, evaluation and curated-prior contracts.

These tests use temporary files and synthetic reports, without launching any
training/controller subprocesses. Repository policy still requires execution
on an authorized compute node.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_subcellular_v6 as evaluator
import prepare_subcellular_v6_lr as preparer
import run_subcellular_v6 as controller
import train_subcellular_v6 as trainer


def prepared_config(route='A'):
    config = yaml.safe_load((ROOT / 'configs/subcellular_lr_v6_20260920.template.yaml').read_text())
    config['model']['route'] = route
    config['lr'].update(path='/synthetic/frozen_lr.csv', sha256='a' * 64,
                        source='synthetic-test-only')
    return config


class V6TrainingContractTests(unittest.TestCase):
    def test_only_explicit_routes_visible_binning_one_epoch_and_one_ce_are_accepted(self):
        for route in ('A', 'B'):
            trainer.validate_config(prepared_config(route))
        changes = (
            ('model', 'route', 'C'), ('model', 'route', 'PREPARATION_REQUIRED'),
            ('model', 'route', None), ('model', 'backbone_kind', 'scgpt_spatial'),
            ('model', 'spatial_enabled', 'true'),
            ('protocol', 'bins', 'pre_mask_quantiles'),
            ('protocol', 'whole_cell_mask', 'unmasked_center'),
            ('protocol', 'readout', 'ordinary_cls_plus_zero_initialized_residual'),
            ('training', 'epochs', 2), ('training', 'epochs', 0),
            ('training', 'epochs', True), ('training', 'backbone_trainable_layers', 1),
            ('loss', 'type', 'masked_bin_cross_entropy_plus_lr_loss'),
            ('loss', 'lr_auxiliary_weight', .1),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key, value=value):
                config = prepared_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    trainer.validate_config(config)

    def test_lr_switches_must_be_explicit_booleans_and_agree(self):
        for enabled in (True, False):
            config = prepared_config()
            config['model']['lr_enabled'] = config['lr']['enabled'] = enabled
            trainer.validate_config(config)
        for model_switch, data_switch in ((True, False), (False, True),
                                          (None, None), (1, 1), ('true', 'true')):
            with self.subTest(model=model_switch, data=data_switch):
                config = prepared_config()
                config['model']['lr_enabled'] = model_switch
                config['lr']['enabled'] = data_switch
                with self.assertRaisesRegex(ValueError, 'LR|lr_'):
                    trainer.validate_config(config)
        config = prepared_config()
        del config['model']['lr_enabled']
        del config['lr']['enabled']
        with self.assertRaises(ValueError):
            trainer.validate_config(config)

    def test_source_fingerprints_include_inherited_v5_implementations(self):
        self.assertEqual(trainer.SOURCE_FILES, evaluator.SOURCE_FILES)
        required = {
            'src/spatial_pretrain/models/subcellular_v6.py',
            'src/spatial_pretrain/models/subcellular_v5.py',
            'src/spatial_pretrain/models/subcellular_v4.py',
            'src/spatial_pretrain/models/subcellular_v2.py',
            'src/spatial_pretrain/data/subcellular_v6.py',
            'src/spatial_pretrain/data/subcellular_v5.py',
            'src/spatial_pretrain/data/subcellular_v3.py',
            'src/spatial_pretrain/data/subcellular_v2.py',
            'src/spatial_pretrain/training/subcellular_v4.py',
        }
        self.assertTrue(required.issubset(trainer.SOURCE_FILES))
        self.assertEqual(len(trainer.SOURCE_FILES), len(set(trainer.SOURCE_FILES)))
        recorded = trainer.source_hash()
        self.assertEqual(evaluator.validate_project_sources(recorded), recorded)
        changed = dict(recorded)
        changed['src/spatial_pretrain/models/subcellular_v5.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source differs'):
            evaluator.validate_project_sources(changed)
        del recorded['src/spatial_pretrain/data/subcellular_v5.py']
        with self.assertRaisesRegex(ValueError, 'dependency hashes'):
            evaluator.validate_project_sources(recorded)

    def test_resume_cannot_switch_route_or_lr_prior_even_if_shapes_match(self):
        config = prepared_config('B')
        architecture = dict(model_type='subcellular_alternating_lr_v6', route='B')
        sources = {'synthetic-source': 'verified'}
        state = dict(
            model_version=6, architecture=trainer.ARCHITECTURE, arch_config=architecture,
            config=copy.deepcopy(config), config_sha256=trainer.config_hash(config),
            source_sha256=sources, micro_batch_size=16, global_batch_size=64,
            phase='frozen_backbone', step=3, epoch=0, next_update=3, complete=False,
            model={}, optimizer={})

        def validate(candidate_config):
            return trainer.validate_resume(state, candidate_config, sources, architecture,
                                           micro_batch=16, updates_per_epoch=10)

        self.assertEqual(validate(config), (3, 0, 3))
        for section, key, value in (('model', 'route', 'A'), ('lr', 'sha256', 'b' * 64),
                                    ('protocol', 'bins', 'pre_mask_quantiles')):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'config mismatch'):
                validate(changed)


class V6EvaluationContractTests(unittest.TestCase):
    def test_checkpoint_version_and_architecture_are_not_inferred_from_tensor_shapes(self):
        valid = dict(model_version=6, architecture=evaluator.ARCHITECTURE,
                     arch_config=dict(route='A', backbone_dim=512))
        evaluator.validate_checkpoint_contract(valid)
        for changes in (dict(model_version=5), dict(model_version=4), dict(model_version=None),
                        dict(architecture='cls_residual_feedback_v5'), dict(arch_config={})):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluator.validate_checkpoint_contract(dict(valid, **changes))

    def test_cls_readout_width_is_independent_of_flattened_compartment_width(self):
        dimensions = evaluator.expected_dimensions(True, compartment_dim=192, backbone_dim=512)
        self.assertEqual(len(dimensions), 10)
        for name in ('base', 'joint', 'intrinsic', 'joint_neighbor_shuffle'):
            self.assertEqual(dimensions[name], 512)
        for name in ('nucleus', 'cytoplasm', 'nucleus_local', 'cytoplasm_local',
                     'nucleus_neighbor_shuffle', 'cytoplasm_neighbor_shuffle'):
            self.assertEqual(dimensions[name], 192)
        for width in (0, -1, 2.5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                evaluator.expected_dimensions(compartment_dim=width)

    def test_step_zero_requires_exact_base_intrinsic_and_joint_identity(self):
        base = np.eye(3, dtype=np.float32)
        features = dict(base=base, intrinsic=base.copy(), joint=base.copy())
        report = evaluator.anchor_comparison(features, 0, spatial_enabled=True)
        self.assertTrue(report['step_zero_identity_verified'])
        self.assertEqual(report['joint_minus_base_max_abs'], 0.)
        for name in ('intrinsic', 'joint'):
            changed = copy.deepcopy(features)
            changed[name][0, 0] = np.nextafter(np.float32(1.), np.float32(2.))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Step-zero'):
                evaluator.anchor_comparison(changed, 0, spatial_enabled=True)
        later = dict(features, joint=base + .1)
        self.assertFalse(evaluator.anchor_comparison(later, 100, spatial_enabled=True)['step_zero_identity_verified'])
        with self.assertRaisesRegex(ValueError, 'No-spatial'):
            evaluator.anchor_comparison(later, 100, spatial_enabled=False)
        later['intrinsic'] = later['joint'].copy()
        evaluator.anchor_comparison(later, 100, spatial_enabled=False)


class V6ControllerContractTests(unittest.TestCase):
    """Synthetic checkpoint/metrics fixture adapted from V4 orchestration QA."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        config = prepared_config('B')
        config['output_dir'] = str(self.directory / 'run')
        # Exercise a compartment width distinct from the 512-d plain CLS.
        config['model']['slots_per_compartment'] = 3
        self.config_path = self.directory / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(config))
        args = argparse.Namespace(config=self.config_path, output_dir=None, resume=False,
                                  training_gpus=2, deadline=10000.,
                                  eval_min_seconds=600., train_min_seconds=120.)
        self.runner = controller.Runner(args)
        checkpoint = self.runner.run / 'checkpoints/step_00000100.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'synthetic V6 checkpoint identity')
        self.checkpoint = checkpoint
        self.metrics = self.runner.run / 'downstream_validation/step_00000100/metrics.json'
        self.metrics.parent.mkdir(parents=True)
        dimensions = evaluator.expected_dimensions(True, compartment_dim=192, backbone_dim=512)
        self.report = dict(
            step=100, model_version=6, architecture=controller.ARCHITECTURE,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            evaluation_version=6, config_sha256=self.runner.fingerprint,
            cohort_sha256=controller.COHORT, cells=5200, canonical_cohort=True,
            test_donors_used=False, backbone_kind='scgpt', neighbor_shuffle=True,
            neighbor_shuffle_seed=815, diagnostic_schema_version=1,
            embedding_dimensions=dimensions, arch_config=dict(backbone_dim=512, route='B'),
            lr_prior=dict(sha256=config['lr']['sha256']),
            masked_validation=dict(enabled=True, requested_minimum_batches=16),
            results=[dict(embedding=name, task=task,
                          fixed_C_result=dict(C=.1 if task == 'cell_type' else .01, val_macro_f1=.4))
                     for name in dimensions for task in ('cell_type', 'region')])

    def write(self, report=None):
        controller.atomic_json(self.metrics, self.report if report is None else report)

    def test_complete_512_dimensional_report_matches_route_prior_and_checkpoint(self):
        self.write()
        self.assertTrue(self.runner.evaluated(100))
        self.checkpoint.write_bytes(b'different checkpoint bytes at the same step')
        with self.assertRaisesRegex(ValueError, 'checkpoint identity mismatch'):
            self.runner.evaluated(100)

    def test_route_prior_and_legacy_readout_width_cannot_be_relabeled(self):
        changes = (
            ('arch_config', 'route', 'A', 'route'),
            ('arch_config', 'backbone_dim', 1024, '512'),
            ('lr_prior', 'sha256', 'b' * 64, 'LR prior'),
            ('embedding_dimensions', 'joint', 384, 'readouts'),
            ('embedding_dimensions', 'joint_neighbor_shuffle', 384, 'readouts'),
        )
        for section, key, value, error in changes:
            changed = copy.deepcopy(self.report)
            changed[section][key] = value
            self.write(changed)
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, error):
                self.runner.evaluated(100)

    def test_missing_metrics_and_incomplete_fixed_probe_set_are_not_completion(self):
        self.assertFalse(self.runner.evaluated(100))
        for mutate in (
                lambda report: report['results'].pop(),
                lambda report: report['results'].append(copy.deepcopy(report['results'][0])),
                lambda report: report.update(model_version=5),
                lambda report: report.update(test_donors_used=True)):
            changed = copy.deepcopy(self.report)
            mutate(changed)
            self.write(changed)
            with self.assertRaises(ValueError):
                self.runner.evaluated(100)


class V6CuratedPriorTests(unittest.TestCase):
    def synthetic_source(self):
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['interaction.ligand', 'interaction.receptor',
                         'interaction.interaction_name_2', 'interaction.annotation'])
        writer.writerows([
            ['VEGFA', 'KDR', 'VEGFA - KDR', 'Secreted Signaling'],
            ['VEGFA', 'KDR', 'VEGFA - KDR', 'Secreted Signaling'],
            ['CDH1', 'CDH1', 'CDH1 - CDH1', 'Cell-Cell Contact'],
            ['TGFB1', 'TGFBR1', 'TGFB1 - (TGFBR1+TGFBR2)', 'Secreted Signaling'],
            ['FN1', 'ITGAV_ITGB3', 'FN1 - ITGAV_ITGB3', 'ECM-Receptor'],
            ['INS', 'INSR', 'INS - INSR', 'Intracellular'],
        ])
        return stream.getvalue().encode()

    def test_complexes_are_excluded_as_whole_records_and_duplicates_have_one_weight(self):
        payload = self.synthetic_source()
        with mock.patch.object(preparer, 'SOURCE_SHA', hashlib.sha256(payload).hexdigest()):
            normalized, manifest = preparer.normalize_prior(payload)
        pairs = list(csv.DictReader(io.StringIO(normalized.decode())))
        self.assertEqual(pairs, [dict(ligand='CDH1', receptor='CDH1', weight='1'),
                                 dict(ligand='VEGFA', receptor='KDR', weight='1')])
        self.assertEqual(manifest['source_rows'], 6)
        self.assertEqual(manifest['unique_directed_pairs'], 2)
        self.assertEqual(manifest['excluded_rows'], dict(complex_or_noncanonical_single_gene=2,
                                                        unsupported_annotation=1))
        self.assertTrue(manifest['excluded_complexes'])
        self.assertFalse(manifest['prediction_only_interactions_added'])
        self.assertEqual(manifest['sha256'], hashlib.sha256(normalized).hexdigest())

    def test_unreviewed_source_bytes_fail_before_normalization(self):
        payload = self.synthetic_source()
        with mock.patch.object(preparer, 'SOURCE_SHA', hashlib.sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                preparer.normalize_prior(payload + b'\n')

    def test_existing_asset_or_config_is_reused_only_when_bytes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'asset.csv'
            preparer.immutable_write(path, b'reviewed existing bytes\n')
            preparer.immutable_write(path, b'reviewed existing bytes\n')
            with self.assertRaisesRegex(FileExistsError, 'Refusing to replace'):
                preparer.immutable_write(path, b'new unreviewed bytes\n')
            self.assertEqual(path.read_bytes(), b'reviewed existing bytes\n')


if __name__ == '__main__':
    unittest.main()
