"""Evaluation correctness tests; run only on an authorized compute node."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_subcellular_v12 import (
    ARCHITECTURE, CANONICAL_COHORT, COUNT_MODES, FIXED_C, PROBE_ROW_NORMALIZATION, READOUTS, TASKS,
    composition_scores, concat_pca, neighbor_composition, normalize_for_probe, project_simplex,
    real_neighbors, representation_diagnostics, validate_completion, joint_count_nll, count_model_inputs,
    aggregate_masked_counts,
)
from watch_subcellular_v12 import checkpoint_step, expected_steps
from evaluate_compartment_velocity_baseline import paired_exposure, predict_train_types, fit_ratios, row_similarity


def completion_fixture():
    dimensions = {name: 1536 if name == 'concat' else 512 for name in READOUTS}
    rows = []
    for name in READOUTS:
        for task in TASKS:
            row = dict(embedding=name, task=task, embedding_dimension=dimensions[name], train_cells=4000, val_cells=1200)
            if task in FIXED_C:
                row['fixed_C_result'] = dict(C=FIXED_C[task], val_macro_f1=.4, val_balanced_accuracy=.3,
                                             solver=dict(converged_gradient=True))
            else:
                row.update(ridge_alpha=1., val_result=dict(mae=.2, js_divergence=.1, js_distance=.3))
            rows.append(row)
    return dict(status='complete', model_version=12, architecture=ARCHITECTURE,
        cohort_sha256=CANONICAL_COHORT, cells=5200, no_test=True, test_donors_used=False,
        sources_verified=True, baseline_separate_pretrained_encoder=True,
        probe_row_normalization=PROBE_ROW_NORMALIZATION,
        train_donors=list(map(str, range(10))), val_donors=['10', '11', '12'],
        fixed_C=FIXED_C, dimensions=dimensions, diagnostics=dict(collapse={name: {} for name in READOUTS}), results=rows,
        masked_count_prediction=dict(both_center_compartments_query_hidden=True, target_counts_passed_to_forward=False,
            results=[dict(split=split, mode=mode, donor='pooled', query_count=100, nll=2.)
                     for split in ('train', 'val') for mode in COUNT_MODES]))


class V12EvaluationTests(unittest.TestCase):
    def test_ten_readouts_includes_all_three_original_baselines(self):
        self.assertEqual(len(READOUTS), 10)
        self.assertEqual(set(READOUTS), {'nucleus', 'cytoplasm', 'local', 'spatial', 'fused', 'concat',
            'concat_pca512', 'scgpt_nucleus', 'scgpt_cytoplasm', 'scgpt_whole'})

    def test_count_forward_excludes_all_target_values(self):
        values = {name: torch.tensor([[1]]) for name in ('nucleus_gene_ids', 'query_gene_ids', 'query_valid_mask',
            'query_nucleus_counts', 'query_cytoplasm_counts', 'query_total_counts')}
        with patch.object(torch.Tensor, 'cuda', lambda tensor, **kwargs: tensor):
            result = count_model_inputs(values)
        self.assertEqual(set(result), {'nucleus_gene_ids', 'query_gene_ids', 'query_valid_mask'})

    def test_joint_likelihood_matches_training_and_simple_distribution(self):
        from spatial_pretrain.training.subcellular_v12 import joint_count_log_prob
        n = torch.tensor([[0., 1., 2., 500.]])
        c = torch.tensor([[0., 1., 3., 750.]])
        parameters = {key: torch.full_like(n, value) for key, value in
            [('mu_total', 2.), ('theta_total', 1.), ('p_nucleus', .5), ('kappa_allocation', 2.)]}
        score = joint_count_nll(n, c, parameters)
        torch.testing.assert_close(score, -joint_count_log_prob(n, c, parameters))
        # theta=1 gives geometric total; alpha=beta=1 gives uniform allocation.
        expected = -torch.log(torch.tensor(1/3, dtype=torch.float64))
        torch.testing.assert_close(score[0, 0], expected)
        expected_two = -torch.log(torch.tensor((1/3)*(2/3)**2/3, dtype=torch.float64))
        torch.testing.assert_close(score[0, 1], expected_two)
        with self.assertRaisesRegex(ValueError, 'nonnegative integers'):
            joint_count_nll(n+.2, c, parameters)

    def test_count_aggregation_excludes_missing_views_without_fabricating_zero_nll(self):
        rows = []
        for queries, nll, patient in [(2, 6., 'a'), (0, 0., 'a'), (4, 4., 'b')]:
            rows.append(dict(split='val', mode='nucleus', patient=patient, query_count=queries,
                positive_query_count=queries, nll_sum=nll, total_abs_error_sum=0., nucleus_abs_error_sum=0.,
                cytoplasm_abs_error_sum=0., allocation_abs_error_sum=0.))
        pooled = next(row for row in aggregate_masked_counts(pd.DataFrame(rows)) if row['donor'] == 'pooled')
        self.assertAlmostEqual(pooled['nll'], 10/6)
        self.assertEqual(pooled['available_cells'], 2)

    def test_completion_requires_true_masked_count_evaluation(self):
        report = completion_fixture(); report['masked_count_prediction']['target_counts_passed_to_forward'] = True
        with self.assertRaisesRegex(ValueError, 'safeguards'):
            validate_completion(report)
        report = completion_fixture(); report['masked_count_prediction']['results'].pop()
        with self.assertRaisesRegex(ValueError, 'count mode'):
            validate_completion(report)

    def test_compartment_exposure_preserves_nuclear_fraction(self):
        n, c = np.array([[2., 8.], [10., 4.]]), np.array([[6., 4.], [2., 8.]])
        nn, cc, exposure, reference = paired_exposure(n, c)
        np.testing.assert_allclose(nn/(nn+cc), n/(n+c))
        np.testing.assert_allclose(nn*exposure[:, None], n)
        self.assertEqual(reference, 22.)

    def test_train_only_celltype_assignment_has_no_validation_label_argument(self):
        n = np.array([[9., 0.], [10., 0.], [0., 8.], [0., 9.], [5., 0.], [0., 5.]])
        train = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        labels, classes, centers, margin = predict_train_types(n, n, train, np.array(['A', 'A', 'B', 'B']))
        np.testing.assert_array_equal(labels, ['A', 'A', 'B', 'B', 'A', 'B'])
        changed = n.copy(); changed[~train] *= 100
        _, _, centers2, _ = predict_train_types(changed, changed, train, np.array(['A', 'A', 'B', 'B']))
        np.testing.assert_array_equal(centers, centers2)
        with self.assertRaisesRegex(ValueError, 'Only training labels'):
            predict_train_types(n, n, train, np.array(['A']*6))

    def test_ratio_fit_separates_celltypes_and_filters_constant_genes(self):
        c = np.column_stack((np.tile(np.arange(1., 51.), 2), np.ones(100)))
        n = c*np.concatenate((np.full(50, 2.), np.full(50, 4.)))[:, None]
        labels = np.array(['A']*50+['B']*50)
        fit = fit_ratios(n, c, labels, np.array(['A', 'B']))
        np.testing.assert_allclose(fit['rho'][:, 0], [2., 4.])
        np.testing.assert_array_equal(fit['reliable'], [[True, False], [True, False]])
        np.testing.assert_allclose(row_similarity(n, n, np.ones_like(n, dtype=bool)), 1.)

    def test_probe_normalization_does_not_modify_raw_readouts(self):
        raw = np.array([[3., 4.], [0., 0.], [-2., 2.]], dtype=np.float32)
        original = raw.copy()
        normalized = normalize_for_probe(raw)
        np.testing.assert_array_equal(raw, original)
        self.assertFalse(np.shares_memory(raw, normalized))
        np.testing.assert_allclose(normalized[0], [.6, .8])
        np.testing.assert_array_equal(normalized[1], [0., 0.])
        np.testing.assert_allclose(np.linalg.norm(normalized[[0, 2]], axis=1), 1.)

    def test_positive_row_scaling_does_not_change_probe_inputs(self):
        raw = np.array([[3., 4., -2.], [1., -4., 0.], [0., 0., 0.]], dtype=np.float32)
        scaled = raw * np.array([[.25], [5.], [10.]], dtype=np.float32)
        np.testing.assert_allclose(normalize_for_probe(raw), normalize_for_probe(scaled), atol=1e-7)

    def test_original_nuclear_baseline_matches_v12_packing_under_truncation(self):
        from spatial_pretrain.data.subcellular_v12 import encode_v12_views
        from spatial_pretrain.data.subcellular_v6_left import encode_whole_cell_v6_left
        genes = np.array([9, 4, 8, 7], dtype=np.int64)
        nuclear = np.array([[3, 4, 5, 6], [1, 0, 1, 2]], dtype=np.uint32)
        cyto = np.array([[0, 2, 0, 1], [2, 3, 1, 0]], dtype=np.uint32)
        view = encode_v12_views(nuclear, cyto, genes, np.array([10, 11]), np.array([[0., 0.], [1., 0.]]),
            pad_token_id=0, cls_token_id=1, neighbors=1, max_tokens=3, query_genes=2,
            seed=42, slice_seed=7, mask=False)
        order = np.argsort(genes, kind='stable')
        baseline = encode_whole_cell_v6_left(nuclear[0, order], np.zeros(len(genes), dtype=np.int64), genes[order],
            pad_token_id=0, cls_token_id=1, max_tokens=3, seed=np.random.SeedSequence([42, 7, 10, 355]))
        np.testing.assert_array_equal(view['nucleus_gene_ids'], baseline['whole_gene_ids'])
        np.testing.assert_array_equal(view['nucleus_bins'], baseline['whole_input_bins'])
        np.testing.assert_array_equal(view['nucleus_valid'], baseline['whole_token_valid_mask'])

    def test_physical_targets_exclude_center_and_report_unknown_coverage(self):
        obs = pd.DataFrame(dict(x_centroid=[0., 1., 0., 2., 100.], y_centroid=[0., 0., 2., 0., 100.],
                                final_CT=['A', None, 'B', 'Unseen', 'A']))
        target, coverage = neighbor_composition(obs, np.array([0, 4]), ['A', 'B'], radius=5.)
        np.testing.assert_array_equal(target, [[0., 1.], [0., 0.]])
        np.testing.assert_array_equal(coverage['total'], [3, 0])
        np.testing.assert_array_equal(coverage['known'], [1, 0])
        np.testing.assert_array_equal(coverage['missing'], [1, 0])
        np.testing.assert_array_equal(coverage['unseen'], [1, 0])
        np.testing.assert_allclose(coverage['coverage'], [1/3, 0])

    def test_neighbor_identity_exclusion_even_when_coordinates_tie(self):
        coords = np.array([[0., 0.], [0., 0.], [1., 0.], [2., 0.]])
        centers = np.array([0, 1, 3])
        result = real_neighbors(coords, centers, 3)
        self.assertEqual(result.shape, (3, 3))
        for center, row in zip(centers, result):
            self.assertNotIn(center, row)
            self.assertEqual(len(np.unique(row)), 3)
        padded = real_neighbors(coords[:2], np.array([0]), 8)
        self.assertEqual(padded[0, 0], 1)
        np.testing.assert_array_equal(padded[0, 1:], -np.ones(7, dtype=int))

    def test_ridge_predictions_project_to_valid_probabilities(self):
        values = np.array([[-3., -2., -1.], [1., 1., 1.], [.2, .3, .5], [100., -10., 0.]])
        result = project_simplex(values)
        self.assertTrue((result >= 0).all())
        np.testing.assert_allclose(result.sum(axis=1), 1.)
        np.testing.assert_allclose(result[1], [1/3, 1/3, 1/3])
        np.testing.assert_allclose(result[2], values[2])
        np.testing.assert_array_equal(result[3], [1., 0., 0.])
        with self.assertRaises(ValueError):
            project_simplex(np.array([[np.nan, 1.]]))

    def test_js_distance_zero_and_disjoint_maximum(self):
        target = np.array([[1., 0.], [.5, .5]])
        self.assertEqual(composition_scores(target, target)['js_distance'], 0.)
        different = composition_scores(np.array([[1., 0.]]), np.array([[0., 1.]]))
        self.assertAlmostEqual(different['js_divergence'], np.log(2))
        self.assertAlmostEqual(different['js_distance'], np.sqrt(np.log(2)))
        with self.assertRaises(ValueError):
            composition_scores(np.array([[0., 0.]]), np.array([[.5, .5]]))

    def test_pca_fit_is_invariant_to_heldout_values(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(30, 8))
        train = np.arange(30) < 20
        first, a = concat_pca(features, train, components=4)
        changed = features.copy(); changed[~train] *= 1000
        second, b = concat_pca(changed, train, components=4)
        np.testing.assert_allclose(first[train], second[train], atol=1e-12)
        self.assertEqual(a['scaler_mean_sha256'], b['scaler_mean_sha256'])
        self.assertEqual(a['components_sha256'], b['components_sha256'])
        self.assertFalse(np.allclose(first[~train], second[~train]))
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            concat_pca(features, train, components=512)

    def test_collapse_diagnostic_detects_constant_features(self):
        obs = pd.DataFrame(dict(split=['train', 'val', 'val', 'val']))
        features = dict(constant=np.ones((4, 2)), variable=np.array([[0., 0.], [1., 0.], [0., 1.], [-1., 0.]]))
        result = representation_diagnostics(obs, features)['collapse']
        self.assertEqual(result['constant']['effective_rank'], 0.)
        self.assertEqual(result['constant']['mean_feature_variance'], 0.)
        self.assertGreater(result['variable']['effective_rank'], 1.)

    def test_completion_requires_every_embedding_every_task(self):
        report = completion_fixture()
        validate_completion(report)
        incomplete = copy.deepcopy(report); incomplete['results'].pop()
        with self.assertRaisesRegex(ValueError, 'all three tasks'):
            validate_completion(incomplete)
        duplicate = copy.deepcopy(report); duplicate['results'][-1] = duplicate['results'][0]
        with self.assertRaisesRegex(ValueError, 'all three tasks'):
            validate_completion(duplicate)
        missing_diagnostic = copy.deepcopy(report); missing_diagnostic['diagnostics']['collapse'].pop('concat_pca512')
        with self.assertRaisesRegex(ValueError, 'diagnostics'):
            validate_completion(missing_diagnostic)

    def test_completion_rejects_protocol_changes_and_unconverged_probes(self):
        for key, value in [('cohort_sha256', 'other'), ('test_donors_used', True), ('sources_verified', False)]:
            report = completion_fixture(); report[key] = value
            with self.assertRaises(ValueError):
                validate_completion(report)
        report = completion_fixture(); report['results'][0]['fixed_C_result']['solver']['converged_gradient'] = False
        with self.assertRaisesRegex(ValueError, 'Unconverged'):
            validate_completion(report)
        report = completion_fixture(); report['dimensions']['concat_pca512'] = 128
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            validate_completion(report)
        report = completion_fixture(); report['results'][2]['ridge_alpha'] = .01
        with self.assertRaisesRegex(ValueError, 'composition probe'):
            validate_completion(report)

    def test_queue_requires_zero_every_interval_and_final(self):
        self.assertEqual(expected_steps(dict(total_updates=250)), {0, 100, 200, 250})
        self.assertEqual(expected_steps(dict(total_updates=200)), {0, 100, 200})
        self.assertEqual(checkpoint_step(Path('step_000100.pt')), 100)
        self.assertIsNone(checkpoint_step(Path('last.pt')))
        self.assertIsNone(checkpoint_step(Path('step_100.partial')))
        self.assertIsNone(checkpoint_step(Path('resume_latest.pt')))
        with self.assertRaises(ValueError):
            expected_steps(dict(total_updates=-1))


if __name__ == '__main__':
    unittest.main()
