"""Synthetic completion evidence; run only on an authorized compute node.

Checkpoint files are tiny byte fixtures, never real model tensors. The tests do
not launch controllers, training, evaluation, or Slurm commands.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_subcellular_v6_epoch as compare
from run_subcellular_v6 import COHORT


class V6ComparisonEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = yaml.safe_load((ROOT / 'configs/subcellular_lr_v6_20260920.template.yaml').read_text())
        self.config['model'].update(route='A', slots_per_compartment=3)
        self.config['lr'].update(path='/synthetic/lr.csv', sha256='a' * 64, source='test-only')
        self.config['output_dir'] = str(self.directory / 'run')
        self.config_path = self.directory / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.fingerprint = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.run = Path(self.config['output_dir'])
        self.run.mkdir()
        self.final_step = 205
        self.steps = (0, 100, 200, self.final_step)
        backbone = dict(backbone_kind='scgpt', weights_sha256='1' * 64, source_sha256='2' * 64,
                        args_sha256='3' * 64, vocab_sha256='4' * 64)
        architecture = dict(route='A', backbone_dim=512, slots_per_compartment=3, slot_dim=64)
        sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in compare.REQUIRED_SOURCE_FILES}
        prior = dict(sha256=self.config['lr']['sha256'], enabled=True)
        optimizer = dict(schema='synthetic_muon_manifest', groups=['hidden', 'aux'])
        self.schedule = dict(
            total_updates=self.final_step, updates_per_epoch=self.final_step, selected_centers=13057,
            global_batch_size=64, selected_center_stride=self.config['data']['center_stride'],
            input_binning=self.config['protocol']['bins'], loss=self.config['loss'],
            backbone_trainable_layers=0, model_version=6, architecture=compare.ARCHITECTURE,
            arch_config=architecture, backbone_provenance=backbone, source_sha256=sources,
            optimizer_manifest=optimizer, lr_prior=prior)
        self.write_json(self.run / 'schedule.json', self.schedule)
        self.progress = dict(step=self.final_step, complete=True, epoch=1, next_update=0,
                             config_sha256=self.fingerprint, model_version=6, architecture=compare.ARCHITECTURE)
        self.write_json(self.run / 'progress.json', self.progress)
        (self.run / 'TRAIN_COMPLETE').write_text(f'step={self.final_step}\n')
        self.status_path = self.run.with_name(self.run.name + '.controller') / 'status.json'
        self.status = dict(status='training_complete', step=self.final_step, total_updates=self.final_step,
                           pending_evaluation_steps=list(self.steps))
        self.write_json(self.status_path, self.status)
        dimensions = {name: 512 if name in ('base', 'intrinsic', 'joint') else 192
                      for name in ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                                   'nucleus_local', 'cytoplasm_local')}
        dimensions.update({f'{name}_neighbor_shuffle': dimensions[name]
                           for name in ('joint', 'nucleus', 'cytoplasm')})
        provenance = dict(
            backbone_kind='scgpt', config_sha256=self.fingerprint, model_version=6,
            architecture=compare.ARCHITECTURE, arch_config=architecture,
            checkpoint_source_sha256=sources, checkpoint_sources_verified=True,
            checkpoint_backbone_provenance=backbone, loaded_pretrained_backbone=backbone,
            checkpoint_provenance_verified=True, frozen_backbone_weights_verified=True,
            lr_prior=prior, optimizer_manifest=optimizer,
            evaluation_model_source_sha256=sources['src/spatial_pretrain/models/subcellular_v6.py'],
            evaluation_data_source_sha256=sources['src/spatial_pretrain/data/subcellular_v6.py'],
            evaluation_loss_source_sha256=sources['src/spatial_pretrain/training/subcellular_v4.py'])
        self.reports = {}
        for step in self.steps:
            checkpoint = self.checkpoint_path(step)
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f'tiny synthetic checkpoint at {step}'.encode())
            self.reports[step] = dict(
                step=step, model_version=6, architecture=compare.ARCHITECTURE, evaluation_version=6,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                config_sha256=self.fingerprint, cohort_sha256=COHORT, cells=5200, cells_per_donor=400,
                canonical_cohort=True, test_donors_used=False, backbone_kind='scgpt',
                neighbor_shuffle=True, neighbor_shuffle_seed=815, diagnostic_schema_version=1,
                arch_config=copy.deepcopy(architecture), embedding_dimensions=dimensions,
                lr_prior=copy.deepcopy(prior), optimizer_manifest=copy.deepcopy(optimizer),
                source_sha256=copy.deepcopy(sources),
                backbone_provenance=dict(copy.deepcopy(provenance), ordinary_scgpt_initial_weights_verified=step == 0),
                cohort_labels_split_sha256='5' * 64, protocol_sha256='6' * 64, vocab_mapping_sha256='7' * 64,
                train_donors=[f'train{index}' for index in range(10)],
                val_donors=['val0', 'val1', 'val2'], evaluation_batch_size=16, seed=42,
                context_cells=64, whole_cell_protocol='fixed visible whole-cell protocol',
                binning_protocol='visible-only test fixture', graph='fixed patch graph',
                anchor_comparison=dict(step_zero_identity_verified=step == 0),
                masked_validation=dict(enabled=True, requested_minimum_batches=16, effective_batches=16,
                    centers=256, total_validation_centers=500, slices_covered=48,
                    sample_indices_sha256='8' * 64, query_targets_sha256='9' * 64,
                    neighbor_shuffle_applied=False, loss_mean=dict(total=2., masked_ce=2.),
                    binning_protocol='visible-only test fixture'),
                results=[dict(embedding=name, task=task, train_cells=4000, val_cells=1200,
                    fixed_C_result=dict(C=.1 if task == 'cell_type' else .01, val_macro_f1=.4,
                        per_donor_macro_f1=dict(val0=.3, val1=.4, val2=.5),
                        solver=dict(converged_gradient=True, gradient_max=1e-5, objective=.2)))
                    for name in dimensions for task in ('cell_type', 'region')])
            self.write_report(step)

    @staticmethod
    def write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def metrics_path(self, step):
        return self.run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'

    def checkpoint_path(self, step):
        return self.run / 'checkpoints' / f'step_{step:08d}.pt'

    def write_report(self, step, payload=None):
        self.write_json(self.metrics_path(step), self.reports[step] if payload is None else payload)

    def read(self):
        return compare.read_run(self.config_path, 'A')

    def test_source_contract_tracks_the_trainer_without_importing_torch(self):
        tree = ast.parse((ROOT / 'scripts/train_subcellular_v6.py').read_text())
        source_assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                                 and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES'
                                         for target in node.targets))
        self.assertEqual(compare.REQUIRED_SOURCE_FILES, ast.literal_eval(source_assignment.value))

    def test_training_only_and_legacy_completion_require_all_evidence(self):
        cfg, schedule, final, initial = self.read()
        self.assertEqual((final['step'], initial['step']), (205, 0))
        self.assertEqual(cfg, self.config)
        self.assertEqual(schedule, self.schedule)
        self.write_json(self.status_path, dict(self.status, status='complete', pending_evaluation_steps=[]))
        self.assertEqual(self.read()[2]['step'], 205)

    def test_missing_each_required_evaluation_including_off_interval_final_fails(self):
        for step in self.steps:
            with self.subTest(step=step):
                self.metrics_path(step).unlink()
                with self.assertRaisesRegex(ValueError, 'Missing required evaluation'):
                    self.read()
                self.write_report(step)

    def test_training_completion_and_empty_pending_list_are_not_evaluation(self):
        self.write_json(self.status_path, dict(self.status, pending_evaluation_steps=[]))
        self.metrics_path(100).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing required evaluation'):
            self.read()

    def test_missing_or_changed_intermediate_checkpoint_fails(self):
        checkpoint = self.checkpoint_path(100)
        checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            self.read()
        checkpoint.write_bytes(b'checkpoint replacement')
        with self.assertRaisesRegex(ValueError, 'checkpoint identity mismatch'):
            self.read()

    def test_every_intermediate_report_requires_complete_twenty_readouts(self):
        for mutation in (lambda report: report['results'].pop(),
                         lambda report: report['results'].append(copy.deepcopy(report['results'][0])),
                         lambda report: report['embedding_dimensions'].update(joint=192)):
            changed = copy.deepcopy(self.reports[100])
            mutation(changed)
            self.write_report(100, changed)
            with self.assertRaisesRegex(ValueError, 'readouts'):
                self.read()

    def test_fixed_probe_scores_donors_and_convergence_are_strict(self):
        mutations = (
            lambda fixed: fixed.update(C=1.),
            lambda fixed: fixed.update(val_macro_f1=float('nan')),
            lambda fixed: fixed.update(val_macro_f1=True),
            lambda fixed: fixed['solver'].update(converged_gradient=False),
            lambda fixed: fixed['solver'].update(gradient_max=.01),
            lambda fixed: fixed['solver'].update(objective=float('nan')),
            lambda fixed: fixed['per_donor_macro_f1'].update(val0=float('inf')),
            lambda fixed: fixed.update(per_donor_macro_f1={}),
        )
        for mutation in mutations:
            changed = copy.deepcopy(self.reports[100])
            mutation(changed['results'][0]['fixed_C_result'])
            self.write_report(100, changed)
            with self.assertRaises(ValueError):
                self.read()

    def test_intermediate_identity_and_schedule_mismatches_fail(self):
        mutations = (
            lambda report: report.update(cells=5199),
            lambda report: report.update(cohort_sha256='0' * 64),
            lambda report: report.update(config_sha256='0' * 64),
            lambda report: report.update(test_donors_used=True),
            lambda report: report['source_sha256'].update({'scripts/train_subcellular_v6.py': '0' * 64}),
            lambda report: report['lr_prior'].update(sha256='0' * 64),
            lambda report: report['optimizer_manifest'].update(schema='foreign'),
            lambda report: report['arch_config'].update(route='B'),
            lambda report: report['backbone_provenance'].update(frozen_backbone_weights_verified=False),
            lambda report: report['backbone_provenance']['loaded_pretrained_backbone'].update(weights_sha256='0' * 64),
            lambda report: report.update(cohort_labels_split_sha256='0' * 64),
            lambda report: report['masked_validation'].update(query_targets_sha256='0' * 64),
            lambda report: report['masked_validation'].update(effective_batches=1),
        )
        for mutation in mutations:
            changed = copy.deepcopy(self.reports[100])
            mutation(changed)
            self.write_report(100, changed)
            with self.assertRaises(ValueError):
                self.read()

    def test_baseline_must_stay_constant_at_intermediate_steps_and_per_donor(self):
        for field in ('aggregate', 'donor'):
            changed = copy.deepcopy(self.reports[100])
            baseline = changed['results'][0]['fixed_C_result']
            if field == 'aggregate':
                baseline['val_macro_f1'] += .01
            else:
                baseline['per_donor_macro_f1']['val0'] += .01
            self.write_report(100, changed)
            with self.assertRaisesRegex(ValueError, 'baseline changed'):
                self.read()

    def test_current_schedule_must_match_completed_progress_and_config(self):
        for key, value in (('global_batch_size', 32), ('selected_centers', 64),
                           ('updates_per_epoch', 100), ('selected_center_stride', 1),
                           ('source_sha256', {})):
            self.write_json(self.run / 'schedule.json', dict(self.schedule, **{key: value}))
            with self.assertRaises(ValueError):
                self.read()

    def test_initial_identity_verification_is_required_after_full_cycle_checks(self):
        changed = copy.deepcopy(self.reports[0])
        changed['anchor_comparison']['step_zero_identity_verified'] = False
        self.write_report(0, changed)
        with self.assertRaisesRegex(ValueError, 'initialization'):
            self.read()


if __name__ == '__main__':
    unittest.main()
