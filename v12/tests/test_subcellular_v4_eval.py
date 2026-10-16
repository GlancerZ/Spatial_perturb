"""CPU-sized V4 evaluator checks; run on a verified compute allocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location('v4_evaluator_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/evaluate_subcellular_v4.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def diagnostic_batch():
    batch = dict(query_compartment_ids=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        query_valid_mask=torch.tensor([[True, True, False], [True, True, False]]),
        target_bins=torch.tensor([[0, 50, -2], [25, 0, -2]]))
    logits = torch.zeros(2, 3, 51)
    # Positive-bin conditional distribution is uniform, hence expected rank=.5.
    logits[0, 0, 0] = 10.
    logits[1, 1, 0] = 10.
    logits[~batch['query_valid_mask']] = torch.nan
    return batch, logits


class SingleHeadDiagnosticTests(unittest.TestCase):
    def test_nonzero_metrics_and_conditional_positive_rank_exclude_padding(self):
        batch, logits = diagnostic_batch()
        result = module.decoder_metrics(logits, batch, batch['query_valid_mask'])
        self.assertEqual(result['queries'], 4)
        self.assertEqual(result['positive_queries'], 2)
        self.assertEqual(result['nonzero_balanced_accuracy'], 1.)
        self.assertEqual(result['nonzero_auroc'], 1.)
        self.assertEqual(result['nonzero_average_precision'], 1.)
        expected = (.25 + (.5 - 24. / 49.) ** 2) / 2.
        self.assertAlmostEqual(result['positive_rank_mse'], expected, places=10)
        self.assertAlmostEqual(result['positive_bin_mae'], 12.5, places=10)
        changed = dict(batch, target_bins=batch['target_bins'].float())
        changed['target_bins'][~batch['query_valid_mask']] = torch.nan
        self.assertEqual(module.decoder_metrics(logits, changed, changed['query_valid_mask']), result)

    def test_positive_rank_conditions_on_positive_bins_even_when_p0_is_large(self):
        batch = dict(target_bins=torch.tensor([[50]]))
        logits = torch.full((1, 1, 51), -1000.)
        logits[..., 0], logits[..., 50] = 1000., 0.
        result = module.decoder_metrics(logits, batch, torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(result['positive_rank_mse'], 0.)
        self.assertAlmostEqual(result['positive_probability_mean'], 0.)
        self.assertTrue(np.isfinite(result['masked_bin_ce_query_mean']))

    def test_ap_and_auroc_keep_order_when_nonzero_probabilities_saturate(self):
        batch = dict(target_bins=torch.tensor([[0, 50]]))
        valid = torch.ones(1, 2, dtype=torch.bool)
        # Both probabilities round to the same endpoint, while the positive
        # example still has strictly greater nonzero evidence than the zero.
        for gaps, endpoint in (((40., 80.), 1.), ((-1000., -900.), 0.)):
            with self.subTest(gaps=gaps):
                logits = torch.full((1, 2, 51), -2000.)
                logits[..., 0] = 0.
                logits[0, :, 50] = torch.tensor(gaps)
                result = module.decoder_metrics(logits, batch, valid)
                self.assertEqual(result['positive_probability_mean'], endpoint)
                self.assertEqual(result['negative_probability_mean'], endpoint)
                self.assertEqual(result['nonzero_average_precision'], 1.)
                self.assertEqual(result['nonzero_auroc'], 1.)
                self.assertTrue(np.isfinite(result['masked_bin_ce_query_mean']))

    def test_aggregate_uses_one_training_loss_and_one_decoder(self):
        batch, logits = diagnostic_batch()
        result = module.aggregate_masked_predictions({'joint_logits': logits}, batch, {}, 100)
        self.assertEqual(set(result['loss_mean']), {'total', 'masked_ce'})
        self.assertEqual(result['loss_mean']['total'], result['loss_mean']['masked_ce'])
        self.assertEqual(set(result['decoders']), {'joint'})
        self.assertEqual(result['decoders']['joint']['queries'], 4)
        self.assertEqual(result['by_query_compartment']['nucleus']['queries'], 2)
        self.assertEqual(result['by_query_compartment']['cytoplasm']['queries'], 2)
        self.assertTrue(np.isfinite(result['loss_mean']['total']))

    def test_empty_query_subset_uses_null_diagnostics(self):
        batch, logits = diagnostic_batch()
        result = module.decoder_metrics(logits, batch, torch.zeros_like(batch['query_valid_mask']))
        self.assertEqual(result['queries'], 0)
        self.assertIsNone(result['masked_bin_ce_query_mean'])
        self.assertIsNone(result['positive_rank_mse'])
        self.assertIsNone(result['nonzero_auroc'])

    def test_wrong_shape_invalid_valid_targets_and_nonfinite_logits_are_rejected(self):
        batch, logits = diagnostic_batch()
        with self.assertRaisesRegex(ValueError, '51'):
            module.decoder_metrics(logits[..., :2], batch, batch['query_valid_mask'])
        batch['target_bins'][0, 0] = -2
        with self.assertRaisesRegex(ValueError, 'integer bins'):
            module.decoder_metrics(logits, batch, batch['query_valid_mask'])
        batch['target_bins'][0, 0] = 0
        logits[0, 0, 1] = torch.nan
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite'):
            module.decoder_metrics(logits, batch, batch['query_valid_mask'])


class ShuffleTests(unittest.TestCase):
    def test_shuffle_changes_expression_at_fixed_positions_without_changing_visibility(self):
        visibility = np.tile([[False, False], [True, False], [False, True], [True, True]], (6, 1))
        nodes = np.stack([np.r_[center, np.delete(np.arange(24), center)[:7]] for center in range(24)])
        original = nodes.copy()
        shuffled = module.shuffle_expression_nodes(nodes, visibility, 71)
        np.testing.assert_array_equal(nodes, original)
        np.testing.assert_array_equal(shuffled[:, 0], nodes[:, 0])
        np.testing.assert_array_equal(visibility[shuffled], visibility[nodes])
        self.assertGreater(np.mean(shuffled[:, 1:] != nodes[:, 1:]), .5)
        for center, row in zip(nodes[:, 0], shuffled):
            self.assertEqual(len(np.unique(row)), len(row))
            self.assertNotIn(center, row[1:])
        # A same-neighbor-list permutation is an ineffective ablation. At least
        # one patch must actually receive a different expression source set.
        self.assertTrue(any(set(before[1:]) != set(after[1:]) for before, after in zip(nodes, shuffled)))
        np.testing.assert_array_equal(shuffled, module.shuffle_expression_nodes(nodes, visibility, 71))
        self.assertFalse(np.array_equal(shuffled, module.shuffle_expression_nodes(nodes, visibility, 72)))

    def test_center_reservation_survives_full_slide_patch_and_singleton_strata(self):
        visibility = np.array([[False, False], [True, True], [True, True], [True, True]])
        nodes = np.stack([np.r_[i, np.delete(np.arange(4), i)] for i in range(4)])
        shuffled = module.shuffle_expression_nodes(nodes, visibility, 1)
        for before, after in zip(nodes, shuffled):
            self.assertEqual(before[0], after[0])
            self.assertEqual(len(np.unique(after)), 4)
        np.testing.assert_array_equal(visibility[nodes], visibility[shuffled])

    def test_duplicates_and_cross_slide_ids_are_rejected(self):
        visibility = np.ones((4, 2), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, 'unique'):
            module.shuffle_expression_nodes([[0, 1, 1]], visibility, 1)
        with self.assertRaisesRegex(ValueError, 'same slide'):
            module.shuffle_expression_nodes([[0, 4]], visibility, 1)

    def test_visibility_uses_only_vocabulary_matched_genes(self):
        nuclear = np.array([[0., 1., 0.], [2., 0., 0.]])
        cytoplasm = np.array([[0., 0., 3.], [0., 5., 0.]])
        actual = module.measured_visibility(nuclear, cytoplasm, np.array([0, 2]))
        np.testing.assert_array_equal(actual, [[False, True], [True, False]])


class ReadoutTests(unittest.TestCase):
    def test_dimensions_follow_slot_capacity_and_include_independent_nuclear_qa(self):
        dims = module.expected_dimensions(True, 24)
        self.assertEqual(dims['joint'], 48)
        self.assertEqual(dims['nucleus'], 24)
        self.assertEqual(dims['nucleus_local'], 24)
        self.assertEqual(dims['nucleus_neighbor_shuffle'], 24)
        self.assertEqual(len(dims), 10)
        self.assertEqual(module.expected_dimensions()['joint'], 1024)

    def test_comparison_uses_fixed_c_and_donor_matched_scores(self):
        results = []
        for name, score in [('nucleus', .6), ('nucleus_local', .5), ('nucleus_neighbor_shuffle', .4)]:
            results.append(dict(embedding=name, task='cell_type',
                fixed_C_result=dict(C=.1, val_macro_f1=score, per_donor_macro_f1={'d1': score}),
                best=dict(C=10, val_macro_f1=.99)))
        compared = module.fixed_probe_comparisons(results)
        self.assertEqual(len(compared), 2)
        self.assertAlmostEqual(compared[0]['val_macro_f1_delta'], .1)
        self.assertAlmostEqual(compared[1]['per_donor_macro_f1_delta']['d1'], .2)
        results[1]['fixed_C_result']['C'] = 1.
        with self.assertRaisesRegex(ValueError, 'different regularization'):
            module.fixed_probe_comparisons(results)


class ProvenanceTests(unittest.TestCase):
    def test_old_checkpoint_cannot_be_relabelled_without_architecture(self):
        for saved in ({}, {'model_version': 3}, {'model_version': 4},
                      {'model_version': 4, 'architecture': 'geneset_feedback_v4'}):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                module.validate_checkpoint_contract(saved)
        valid = dict(model_version=4, architecture='geneset_feedback_v4', arch_config={'slots_per_compartment': 8})
        module.validate_checkpoint_contract(valid)
        with patch.object(module.torch, 'load', return_value={'model_version': 3}), \
                patch.object(module, 'SubcellularV4') as constructor:
            with self.assertRaisesRegex(ValueError, 'V3 is not compatible'):
                module.load_model(Path('unused.pt'))
            constructor.assert_not_called()

    def test_changed_arch_config_is_rejected_before_weights_or_cuda(self):
        class FakeModel:
            def __init__(self, asset_dir, backbone_kind='scgpt', checkpointing=False):
                self.arch_config = {'slots_per_compartment': 8}

        cfg = {'assets_dir': '/unused', 'model': {'backbone_kind': 'scgpt'}}
        saved = dict(model_version=4, architecture='geneset_feedback_v4',
            arch_config={'slots_per_compartment': 9}, step=100, config=cfg,
            config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
        with patch.object(module.torch, 'load', return_value=saved), \
                patch.object(module, 'validate_project_sources', return_value={}), \
                patch.object(module, 'SubcellularV4', FakeModel):
            with self.assertRaisesRegex(ValueError, 'arch_config'):
                module.load_model(Path('unused.pt'))

    def test_exact_dependency_set_and_content_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in module.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            recorded = {name: module.file_sha256(root / name) for name in module.SOURCE_FILES}
            with patch.object(module, 'ROOT', root):
                self.assertEqual(module.validate_project_sources(recorded), recorded)
                missing = dict(recorded)
                missing.pop(module.SOURCE_FILES[-1])
                with self.assertRaisesRegex(ValueError, 'exact eight'):
                    module.validate_project_sources(missing)
                (root / module.SOURCE_FILES[-1]).write_text('changed dependency')
                with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
                    module.validate_project_sources(recorded)

    def test_output_identity_allows_resume_but_refuses_other_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            module.reserve_output(root, {'checkpoint': 'one'})
            module.reserve_output(root, {'checkpoint': 'one'})
            with self.assertRaisesRegex(ValueError, 'different evaluation'):
                module.reserve_output(root, {'checkpoint': 'two'})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'existing_user_output.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                module.reserve_output(root, {'checkpoint': 'one'})
            self.assertEqual((root / 'existing_user_output.txt').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
