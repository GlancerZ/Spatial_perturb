"""Pure-logic tests of the untrained neighbor controls; CPU, synthetic arrays only."""
from pathlib import Path
import copy
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_subcellular_v12_neighbor_controls as ctl
from evaluate_subcellular_v12 import CANONICAL_COHORT, FIXED_C, TASKS


def synthetic_slice(cells=80, genes=10, extent=800., seed=3):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, extent, size=(cells, 2)).astype(np.float32)
    nuclear = rng.poisson(1., size=(cells, genes)).astype(np.int64)
    cyto = rng.poisson(1., size=(cells, genes)).astype(np.int64)
    nuclear[::5], cyto[::5] = 0, 0
    nuclear[1::5, 0] += 1
    expressed = (nuclear > 0).any(axis=1) | (cyto > 0).any(axis=1)
    return coords, nuclear, cyto, np.arange(2, 2 + genes, dtype=np.int64), expressed


def report_fixture():
    dimensions = {name: 1024 if name.endswith('_plus_nbr') else 343 if name == 'nbr_expr_mean' else 512 for name in ctl.READOUTS}
    rows, scores = [], {}
    for name in ctl.READOUTS:
        for task in TASKS:
            row = dict(embedding=name, task=task, embedding_dimension=dimensions[name], train_cells=4000, val_cells=1200)
            value = ctl.ANCHORS.get((name, task), .3)
            if task in FIXED_C:
                row['fixed_C_result'] = dict(C=FIXED_C[task], val_macro_f1=value, val_balanced_accuracy=.3,
                    per_donor_macro_f1=dict(a=.2, b=.4), solver=dict(converged_gradient=True))
            else:
                row.update(ridge_alpha=1., mean_val_label_coverage=.9, val_result=dict(mae=.1, js_divergence=.1, js_distance=value))
            rows.append(row); scores[(name, task)] = ctl.score_of(row)
    return dict(status='complete', experiment=ctl.EXPERIMENT, cohort_sha256=CANONICAL_COHORT, cells=5200,
        test_donors_used=False, trained_parameters=0, fixed_C=FIXED_C, train_donors=list(map(str, range(10))),
        val_donors=['10', '11', '12'], dimensions=dimensions, results=rows, anchor_check=ctl.anchor_check(scores),
        anchor_mismatch_allowed=False, encoder_path_check=dict(samples=13, failed_samples=[]),
        encoder_path_mismatch_allowed=False, comparison_hint=ctl.comparison_hint(scores, dimensions)), scores


class ReadoutContractTests(unittest.TestCase):
    def test_requested_readouts_and_anchor_order(self):
        self.assertEqual(ctl.READOUTS[:2], ('scgpt_whole', 'scgpt_nucleus'))
        self.assertEqual(len(set(ctl.READOUTS)), len(ctl.READOUTS))
        self.assertTrue({'nbr_mean', 'nbr_mean_dw', 'nbr_mean_shuffled', 'nbr_expr_mean', 'whole_plus_nbr',
            'whole_plus_nbr_pca512', 'nucleus_plus_nbr', 'nucleus_plus_nbr_pca512', 'whole_add_nbr'} <= set(ctl.READOUTS))
        self.assertEqual(set(ctl.ANCHORS), {(name, task) for name in ctl.ANCHOR_READOUTS for task in TASKS})
        self.assertEqual(ctl.ANCHORS[('scgpt_whole', 'region')], .280141)
        self.assertEqual(ctl.V12_REFERENCE[3400]['fused']['region'], .373513)
        self.assertEqual(ctl.V12_REFERENCE[0]['spatial']['region'], .399528)


class NeighborSelectionTests(unittest.TestCase):
    coords = np.array([[0., 0.], [10., 0.], [20., 0.], [150., 0.], [0., 30.]], dtype=np.float32)
    expressed = np.array([True, True, False, True, True])

    def test_center_radius_and_unexpressed_exclusions(self):
        rows, distance, dropped = ctl.model_input_neighbors(self.coords, np.array([0, 3]), self.expressed, 8, 100.)
        np.testing.assert_array_equal(rows[0], [1, 4, -1, -1, -1, -1, -1, -1])
        np.testing.assert_allclose(distance[0, :2], [10., 30.])
        self.assertTrue(np.isinf(distance[0, 2:]).all())
        # The far center has only out-of-radius cells: zero neighbors.
        np.testing.assert_array_equal(rows[1], -np.ones(8, dtype=int))
        self.assertEqual(dropped, dict(candidates=8, beyond_data_radius=5, beyond_model_radius=0, unexpressed=1))
        self.assertFalse((rows == np.array([0, 3])[:, None]).any())

    def test_model_side_radius_is_applied_after_data_radius(self):
        rows, _, dropped = ctl.model_input_neighbors(self.coords, np.array([0]), self.expressed, 8, 100., model_radius=25.)
        np.testing.assert_array_equal(rows[0, :2], [1, -1])
        self.assertEqual(dropped['beyond_model_radius'], 1)
        rows, _, _ = ctl.model_input_neighbors(self.coords, np.array([0]), self.expressed, 8, None)
        np.testing.assert_array_equal(rows[0, :3], [1, 4, 3])

    def test_dropped_neighbor_is_never_replaced_by_the_next_nearest(self):
        rows, _, _ = ctl.model_input_neighbors(self.coords, np.array([4]), self.expressed, 1, 100.)
        np.testing.assert_array_equal(rows, [[0]])
        expressed = self.expressed.copy(); expressed[0] = False
        rows, _, dropped = ctl.model_input_neighbors(self.coords, np.array([4]), expressed, 1, 100.)
        np.testing.assert_array_equal(rows, [[-1]])
        self.assertEqual(dropped['unexpressed'], 1)

    def test_matches_reference_v12_dataset_path_cells_and_tokens(self):
        coords, nuclear, cyto, genes, expressed = synthetic_slice()
        centers = np.flatnonzero(expressed)[:20]
        for model_radius in (100., 60.):
            rows, distance, dropped = ctl.model_input_neighbors(coords, centers, expressed, 8, 100., model_radius)
            self.assertGreater(dropped['beyond_data_radius'], 0)
            self.assertGreater(dropped['unexpressed'], 0)
            self.assertTrue((distance[rows >= 0] <= 100.).all())
            result = ctl.verify_against_v12_inputs(nuclear, cyto, genes, coords, centers, rows, k=8, max_tokens=512,
                query_genes=4, data_radius=100., model_radius=model_radius, pad_token_id=0, cls_token_id=1,
                slice_seed=7, slice_index=0, chunk=6)
            self.assertEqual(result, dict(checked_centers=20, checked_neighbor_token_rows=int((rows >= 0).sum())))
        self.assertGreater(dropped['beyond_model_radius'], 0)
        tampered = rows.copy()
        index = int(np.flatnonzero((rows >= 0).any(axis=1))[0])
        tampered[index] = -1
        with self.assertRaisesRegex(ValueError, 'Neighbor set differs'):
            ctl.verify_against_v12_inputs(nuclear, cyto, genes, coords, centers, tampered, k=8, max_tokens=512,
                query_genes=4, data_radius=100., model_radius=60., pad_token_id=0, cls_token_id=1,
                slice_seed=7, slice_index=0)

    def test_whole_cell_tokens_depend_only_on_summed_counts(self):
        _, nuclear, cyto, genes, expressed = synthetic_slice()
        row = int(np.flatnonzero(expressed)[3])
        zero = np.zeros(len(genes), dtype=np.int64)
        for max_tokens in (512, 3):
            first = ctl.tokenize_whole(nuclear[row], cyto[row], genes, row, 7, max_tokens, 0, 1)
            second = ctl.tokenize_whole(zero, nuclear[row] + cyto[row], genes, row, 7, max_tokens, 0, 1)
            for key in ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'):
                np.testing.assert_array_equal(first[key], second[key])
            self.assertEqual(first['whole_gene_ids'][0], 1)
            self.assertLessEqual(len(first['whole_gene_ids']), max_tokens)

    def test_table_distances(self):
        result = ctl.table_distances(np.array([[0., 0.], [3., 4.]]), np.array([0]), np.array([[1, -1]]))
        self.assertEqual(result[0, 0], 5.)
        self.assertTrue(np.isinf(result[0, 1]))


class ShuffleTests(unittest.TestCase):
    def test_same_count_same_sample_no_center_no_ineligible(self):
        rows = np.array([[1, 2, -1], [-1, -1, -1], [0, -1, -1]])
        centers = np.array([0, 3, 4])
        eligible = np.array([True, True, True, True, True, False])
        result = ctl.shuffled_neighbors(rows, centers, eligible, 42, 9)
        np.testing.assert_array_equal((result >= 0).sum(axis=1), (rows >= 0).sum(axis=1))
        np.testing.assert_array_equal(result[1], [-1, -1, -1])
        for center, row in zip(centers, result):
            row = row[row >= 0]
            self.assertNotIn(center, row)
            self.assertNotIn(5, row)
            self.assertEqual(len(np.unique(row)), len(row))
            self.assertTrue(((row >= 0) & (row < 5)).all())

    def test_deterministic_and_keyed_by_center_not_order(self):
        rows = np.array([[1, 2, -1], [-1, -1, -1], [0, -1, -1]])
        centers = np.array([0, 3, 4])
        eligible = np.ones(60, dtype=bool)
        first = ctl.shuffled_neighbors(rows, centers, eligible, 42, 9)
        np.testing.assert_array_equal(first, ctl.shuffled_neighbors(rows, centers, eligible, 42, 9))
        order = np.array([2, 0, 1])
        np.testing.assert_array_equal(first[order], ctl.shuffled_neighbors(rows[order], centers[order], eligible, 42, 9))

    def test_seed_changes_draw_and_small_pool_is_rejected(self):
        rows = np.arange(1, 9)[None]
        eligible = np.ones(500, dtype=bool)
        first = ctl.shuffled_neighbors(rows, np.array([0]), eligible, 42, 9)
        self.assertFalse(np.array_equal(first, ctl.shuffled_neighbors(rows, np.array([0]), eligible, 43, 9)))
        self.assertFalse(np.array_equal(first, ctl.shuffled_neighbors(rows, np.array([0]), eligible, 42, 10)))
        with self.assertRaisesRegex(ValueError, 'Too few'):
            ctl.shuffled_neighbors(np.array([[1, 2]]), np.array([0]), np.array([True, True, False]), 42, 9)


class AggregationTests(unittest.TestCase):
    unique = np.array([2, 5, 9])
    encoded = np.array([[1., 0.], [0., 2.], [4., 4.]], dtype=np.float32)
    rows = np.array([[2, 5, -1], [9, -1, -1], [-1, -1, -1]])

    def test_unweighted_mean_and_zero_vector_without_neighbors(self):
        result = ctl.aggregate_neighbors(self.encoded, self.unique, self.rows)
        np.testing.assert_allclose(result, [[.5, 1.], [4., 4.], [0., 0.]])
        self.assertEqual(result.dtype, np.float32)
        empty = ctl.aggregate_neighbors(np.zeros((0, 2)), np.array([], dtype=np.int64), -np.ones((2, 3), dtype=np.int64))
        np.testing.assert_array_equal(empty, np.zeros((2, 2)))

    def test_weights_are_normalized_and_padding_weights_ignored(self):
        weights = np.array([[1., 3., 7.], [2., 0., 0.], [5., 5., 5.]])
        result = ctl.aggregate_neighbors(self.encoded, self.unique, self.rows, weights)
        np.testing.assert_allclose(result, [[.25, 1.5], [4., 4.], [0., 0.]])
        with self.assertRaisesRegex(ValueError, 'positive total weight'):
            ctl.aggregate_neighbors(self.encoded, self.unique, self.rows, np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            ctl.aggregate_neighbors(self.encoded, self.unique, self.rows, -weights)

    def test_distance_weighted_mean_matches_hand_computation(self):
        distance = np.array([[0., 50., np.inf], [100., np.inf, np.inf], [np.inf, np.inf, np.inf]])
        weights = ctl.distance_weights(distance, 50.)
        np.testing.assert_allclose(weights[0], [1., np.exp(-1.), 0.])
        result = ctl.aggregate_neighbors(self.encoded, self.unique, self.rows, weights)
        expected = (np.array([1., 0.]) + np.exp(-1.) * np.array([0., 2.])) / (1 + np.exp(-1.))
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)
        np.testing.assert_allclose(result[1], [4., 4.])
        with self.assertRaises(ValueError):
            ctl.distance_weights(distance, 0.)

    def test_missing_or_unsorted_encoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no encoded row'):
            ctl.aggregate_neighbors(self.encoded, self.unique, np.array([[3, -1, -1]]))
        with self.assertRaisesRegex(ValueError, 'no encoded row'):
            ctl.aggregate_neighbors(self.encoded, self.unique, np.array([[10, -1, -1]]))
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            ctl.aggregate_neighbors(self.encoded, np.array([5, 2, 9]), self.rows)

    def test_center_expression_cannot_enter_its_own_aggregate(self):
        unique = np.array([0, 1, 2])
        encoded = np.array([[9., 9.], [1., 0.], [0., 1.]])
        rows = np.array([[1, 2]])
        changed = encoded.copy(); changed[0] = -100.
        np.testing.assert_array_equal(ctl.aggregate_neighbors(encoded, unique, rows),
                                      ctl.aggregate_neighbors(changed, unique, rows))

    def test_pooled_expression_matches_hand_computation(self):
        counts = np.array([[1, 0, 3], [1, 2, 0]])
        rows = np.array([[1, 4], [4, -1], [-1, -1]])
        result = ctl.pooled_log_expression(counts, np.array([1, 4]), rows)
        np.testing.assert_allclose(result[0], np.log1p(1e4 * np.array([2, 2, 3]) / 7), rtol=1e-6)
        np.testing.assert_allclose(result[1], np.log1p(1e4 * np.array([1, 2, 0]) / 3), rtol=1e-6)
        np.testing.assert_array_equal(result[2], [0., 0., 0.])
        # Pooling raw counts first is not the mean of per-cell normalized profiles.
        separate = np.log1p(1e4 * (np.array([1, 0, 3]) / 4 + np.array([1, 2, 0]) / 3) / 2)
        self.assertFalse(np.allclose(result[0], separate))
        with self.assertRaises(ValueError):
            ctl.pooled_log_expression(-counts, np.array([1, 4]), rows)

    def test_common_panel_is_sorted_intersection(self):
        panels = [np.array(['B', 'A', 'C']), np.array(['C', 'B', 'D'])]
        genes = ctl.common_panel(panels)
        np.testing.assert_array_equal(genes, ['B', 'C'])
        np.testing.assert_array_equal(ctl.panel_columns(panels[0], genes), [0, 2])
        np.testing.assert_array_equal(ctl.panel_columns(panels[1], genes), [1, 0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ctl.common_panel([np.array(['A', 'A'])])
        with self.assertRaisesRegex(ValueError, 'No commonly'):
            ctl.common_panel([np.array(['A']), np.array(['B'])])


class FusionTests(unittest.TestCase):
    def test_concat_and_additive_fusion_use_unit_parts(self):
        first = np.array([[3., 4.], [0., 5.]])
        second = np.array([[0., 2.], [0., 0.]])
        np.testing.assert_allclose(ctl.fuse(first, second, 'concat'), [[.6, .8, 0., 1.], [0., 1., 0., 0.]], atol=1e-7)
        # A center without neighbors keeps exactly its own unit vector.
        np.testing.assert_allclose(ctl.fuse(first, second, 'add'), [[.6, 1.8], [0., 1.]], atol=1e-7)
        np.testing.assert_allclose(ctl.fuse(first * 10, second * .1, 'add'), ctl.fuse(first, second, 'add'), atol=1e-6)
        with self.assertRaises(ValueError):
            ctl.fuse(first, np.zeros((2, 3)), 'add')
        with self.assertRaises(ValueError):
            ctl.fuse(first, second[:1], 'concat')
        with self.assertRaises(ValueError):
            ctl.fuse(first, second, 'product')

    def test_fusion_readouts_shapes_and_train_only_pca(self):
        rng = np.random.default_rng(5)
        features = {name: rng.normal(size=(40, 6)) for name in ('scgpt_whole', 'scgpt_nucleus', 'nbr_mean', 'nbr_mean_shuffled')}
        train = np.arange(40) < 30
        values, pca = ctl.fusion_readouts(features, train, components=4)
        self.assertEqual(set(values), set(ctl.FUSION_READOUTS))
        self.assertEqual(values['whole_plus_nbr'].shape, (40, 12))
        self.assertEqual(values['nucleus_plus_nbr_pca512'].shape, (40, 4))
        self.assertEqual(values['whole_add_nbr'].shape, (40, 6))
        self.assertEqual(set(pca), {'whole_plus_nbr_pca512', 'nucleus_plus_nbr_pca512'})
        self.assertEqual(pca['whole_plus_nbr_pca512']['fit_split'], 'train')
        np.testing.assert_allclose(np.linalg.norm(values['whole_plus_nbr'][:, :6], axis=1), 1., atol=1e-5)
        changed = {name: value.copy() for name, value in features.items()}
        for value in changed.values():
            value[~train] = rng.normal(size=(10, 6)) * 50
        second, other = ctl.fusion_readouts(changed, train, components=4)
        np.testing.assert_allclose(values['whole_plus_nbr_pca512'][train], second['whole_plus_nbr_pca512'][train], atol=1e-5)
        self.assertEqual(pca['whole_plus_nbr_pca512']['components_sha256'], other['whole_plus_nbr_pca512']['components_sha256'])
        self.assertFalse(np.allclose(values['whole_add_nbr'], values['whole_add_nbr_shuffled']))


class ReportingTests(unittest.TestCase):
    def test_anchor_check_tolerance(self):
        scores = {key: value + 1e-3 for key, value in ctl.ANCHORS.items()}
        check = ctl.anchor_check(scores)
        self.assertTrue(check['passed'])
        self.assertAlmostEqual(check['max_abs_difference'], 1e-3)
        self.assertEqual(len(check['rows']), 6)
        scores[('scgpt_nucleus', 'region')] = ctl.ANCHORS[('scgpt_nucleus', 'region')] - 3e-3
        check = ctl.anchor_check(scores)
        self.assertFalse(check['passed'])
        bad = [row for row in check['rows'] if not row['within_tolerance']]
        self.assertEqual([(row['embedding'], row['task']) for row in bad], [('scgpt_nucleus', 'region')])
        self.assertAlmostEqual(bad[0]['difference'], -3e-3)
        self.assertTrue(ctl.anchor_check(scores, tolerance=5e-3)['passed'])
        scores.pop(('scgpt_whole', 'cell_type'))
        self.assertFalse(ctl.anchor_check(scores, tolerance=1.)['passed'])
        json.dumps(check, allow_nan=False)

    def test_anchor_check_states_where_the_anchor_embeddings_came_from(self):
        self.assertIsNone(ctl.anchor_check(dict(ctl.ANCHORS))['scope'])
        source = dict(official_cache_read_only=12, evaluator_path_private_cache=1)
        check = ctl.anchor_check(dict(ctl.ANCHORS), source=source)
        self.assertTrue(check['passed'])
        self.assertEqual(check['embedding_source'], source)
        self.assertIn('12 sample(s) are the official evaluator cache arrays', check['scope'])
        self.assertIn('1 sample(s) were recomputed', check['scope'])
        self.assertIn('encoder_path_check', check['scope'])
        json.dumps(check, allow_nan=False)

    def test_score_and_flat_row_follow_reference_summary_columns(self):
        report, _ = report_fixture()
        common = dict(variant=ctl.EXPERIMENT, step=None, checkpoint_sha256=None, config_sha256='c', cohort_sha256='h', metrics_path='m')
        classification = next(row for row in report['results'] if row['task'] == 'region' and row['embedding'] == 'nbr_mean')
        row = ctl.flat_row(classification, common)
        self.assertEqual(list(row), ['variant', 'step', 'checkpoint_sha256', 'config_sha256', 'cohort_sha256', 'metrics_path',
            'embedding', 'task', 'dimension', 'train_cells', 'val_cells', 'C', 'val_macro_f1', 'val_balanced_accuracy',
            'donor_macro_f1_a', 'donor_macro_f1_b'])
        self.assertEqual((row['C'], row['dimension'], row['val_macro_f1']), (.01, 512, .3))
        composition = next(row for row in report['results'] if row['task'] == 'neighbor_composition' and row['embedding'] == 'scgpt_whole')
        row = ctl.flat_row(composition, common)
        self.assertEqual(list(row)[-5:], ['val_mae', 'val_js_divergence', 'val_js_distance', 'ridge_alpha', 'mean_val_label_coverage'])
        self.assertEqual(ctl.score_of(composition), .381156)
        self.assertEqual(ctl.score_of(classification), .3)

    def test_comparison_hint_lists_controls_beside_v12_reference(self):
        report, scores = report_fixture()
        scores[('nbr_mean', 'region')], scores[('nbr_mean_shuffled', 'region')] = .41, .30
        scores[('whole_add_nbr', 'region')] = .36
        hint = ctl.comparison_hint(scores, report['dimensions'])
        region = hint['region']
        self.assertEqual([row['readout'] for row in region['controls']], list(ctl.READOUTS))
        reference = {(row['step'], row['readout']): row['val_macro_f1'] for row in region['v12_reference']}
        self.assertEqual(reference[(0, 'spatial')], .399528)
        self.assertEqual(reference[(3400, 'spatial')], .391389)
        self.assertEqual(reference[(3400, 'fused')], .373513)
        self.assertAlmostEqual(region['deltas']['nbr_mean_minus_nbr_mean_shuffled'], .11)
        self.assertAlmostEqual(region['deltas']['nbr_mean_minus_v12_spatial_step3400'], .41 - .391389)
        self.assertAlmostEqual(region['deltas']['whole_add_nbr_minus_v12_fused_step3400'], .36 - .373513)
        cell = {(row['step'], row['readout']): row['val_macro_f1'] for row in hint['cell_type']['v12_reference']}
        self.assertEqual(cell[(3400, 'local')], .588161)
        self.assertEqual(hint['v12_masked_count_reference']['nll'], .6006)
        self.assertEqual(len(hint['neighbor_composition']['controls']), len(ctl.READOUTS))
        json.dumps(hint, allow_nan=False)
        partial = ctl.comparison_hint({('nbr_mean', 'region'): .4}, {})
        self.assertNotIn('nbr_mean_minus_nbr_mean_shuffled', partial['region']['deltas'])
        self.assertEqual(partial['region']['per_val_donor_macro_f1_deltas'], {})

    def test_comparison_hint_per_donor_deltas_and_wording(self):
        report, scores = report_fixture()
        per_donor = {('nbr_mean', 'region'): dict(a=.5, b=.3, c=.4), ('nbr_mean_shuffled', 'region'): dict(a=.2, b=.35, c=.4),
                     ('scgpt_whole', 'region'): dict(a=.1, b=.1), ('nbr_mean', 'cell_type'): dict(a=.2)}
        hint = ctl.comparison_hint(scores, report['dimensions'], per_donor)
        deltas = hint['region']['per_val_donor_macro_f1_deltas']
        self.assertEqual(list(deltas['nbr_mean_minus_nbr_mean_shuffled']), ['a', 'b', 'c'])
        np.testing.assert_allclose(list(deltas['nbr_mean_minus_nbr_mean_shuffled'].values()), [.3, -.05, 0.], atol=1e-12)
        # Only donors scored for both readouts are compared.
        np.testing.assert_allclose(list(deltas['nbr_mean_minus_scgpt_whole'].values()), [.4, .2], atol=1e-12)
        self.assertNotIn('whole_add_nbr_minus_whole_add_nbr_shuffled', deltas)
        self.assertEqual(hint['cell_type']['per_val_donor_macro_f1_deltas'], {})
        guide = ' '.join(hint['reading_guide'])
        self.assertNotIn('sample/donor identity', guide)
        self.assertIn('not donor-identity leakage', guide)
        self.assertIn('no evidence that physical locality adds information', guide)
        self.assertIn('zero-neighbor centers stay zero vectors', guide)
        json.dumps(hint, allow_nan=False)

    def test_report_validation(self):
        report, _ = report_fixture()
        ctl.validate_report(report)
        for key, value in [('cohort_sha256', 'other'), ('test_donors_used', True), ('trained_parameters', 5), ('status', 'running')]:
            changed = copy.deepcopy(report); changed[key] = value
            with self.assertRaises(ValueError):
                ctl.validate_report(changed)
        missing = copy.deepcopy(report); missing['results'].pop()
        with self.assertRaisesRegex(ValueError, 'all three tasks'):
            ctl.validate_report(missing)
        unconverged = copy.deepcopy(report); unconverged['results'][0]['fixed_C_result']['solver']['converged_gradient'] = False
        with self.assertRaisesRegex(ValueError, 'Unconverged'):
            ctl.validate_report(unconverged)
        failed = copy.deepcopy(report); failed['anchor_check']['passed'] = False
        with self.assertRaisesRegex(ValueError, 'Anchor'):
            ctl.validate_report(failed)
        failed['anchor_mismatch_allowed'] = True
        ctl.validate_report(failed)
        unverified = copy.deepcopy(report); unverified['encoder_path_check']['failed_samples'] = ['S1']
        with self.assertRaisesRegex(ValueError, 'encoder path'):
            ctl.validate_report(unverified)
        # The anchor override does not waive the encoder path check; it has its own switch.
        unverified['anchor_mismatch_allowed'] = True
        with self.assertRaisesRegex(ValueError, 'encoder path'):
            ctl.validate_report(unverified)
        unverified['encoder_path_mismatch_allowed'] = True
        ctl.validate_report(unverified)
        partial = copy.deepcopy(report); partial['encoder_path_check']['samples'] = 12
        with self.assertRaisesRegex(ValueError, 'encoder path'):
            ctl.validate_report(partial)

    def test_csv_is_atomic_and_blank_for_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            ctl.write_csv(path, [dict(embedding='a', step=None, val_macro_f1=.5), dict(embedding='b', val_js_distance=.2)])
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()), ['metrics.csv'])
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], dict(embedding='a', step='', val_macro_f1='0.5', val_js_distance=''))
            self.assertEqual(rows[1]['val_js_distance'], '0.2')

    def test_pack_tokens_pads_like_reference_baseline(self):
        packed = [dict(whole_gene_ids=np.array([1, 7, 8]), whole_input_bins=np.array([-2., 3., 5.], dtype=np.float32),
                       whole_token_valid_mask=np.ones(3, dtype=bool)),
                  dict(whole_gene_ids=np.array([1]), whole_input_bins=np.array([-2.], dtype=np.float32),
                       whole_token_valid_mask=np.ones(1, dtype=bool))]
        token, bins, valid = ctl.pack_tokens(packed, 0)
        np.testing.assert_array_equal(token, [[1, 7, 8], [1, 0, 0]])
        np.testing.assert_array_equal(bins, [[-2., 3., 5.], [-2., -2., -2.]])
        np.testing.assert_array_equal(valid, [[True, True, True], [True, False, False]])


class EncoderPathCheckTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        # A large shared offset mimics raw scGPT CLS: every pair of cells has a high raw cosine.
        self.reference = rng.normal(size=(40, 16)) + 20.
        self.own = self.reference + rng.normal(scale=.01, size=self.reference.shape)

    def test_aligned_encoding_passes_far_above_the_wrong_row_null(self):
        check = ctl.encoder_path_check(self.own, self.reference)
        self.assertTrue(check['passed'])
        self.assertEqual((check['cells'], check['top1_fraction'], check['rows_below_tolerance']), (40, 1., 0))
        self.assertGreater(check['min_margin'], 0)
        self.assertGreater(check['matched_centered_cosine']['min'], check['null_centered_cosine']['max'])
        self.assertEqual(check['null_centered_cosine']['pairs'], 40 * 39)
        self.assertTrue(check['allclose_atol_0p03_rtol_0p02'])
        json.dumps(check, allow_nan=False)

    def test_raw_cosine_cannot_see_a_row_shift_but_the_null_calibrated_check_does(self):
        shifted = np.roll(self.own, 1, axis=0)
        check = ctl.encoder_path_check(shifted, self.reference)
        self.assertGreater(check['min_raw_cosine'], .98)
        self.assertFalse(check['passed'])
        self.assertEqual(check['top1_fraction'], 0.)
        self.assertEqual(check['rows_below_tolerance'], 40)

    def test_two_swapped_rows_fail_even_with_high_top1_fraction(self):
        swapped = self.own.copy(); swapped[[3, 17]] = swapped[[17, 3]]
        check = ctl.encoder_path_check(swapped, self.reference)
        self.assertEqual(check['top1_fraction'], 38 / 40)
        self.assertGreaterEqual(check['top1_fraction'], ctl.PATH_MIN_TOP1)
        self.assertEqual(check['rows_below_tolerance'], 2)
        self.assertFalse(check['passed'])

    def test_identical_input_cells_are_ties_not_failures(self):
        reference = self.reference.copy(); reference[5] = reference[4]
        own = reference + np.random.default_rng(2).normal(scale=1e-3, size=reference.shape)
        check = ctl.encoder_path_check(own, reference)
        self.assertTrue(check['passed'])
        self.assertGreater(check['null_centered_cosine']['max'], .999)
        self.assertLess(abs(check['min_margin']), ctl.PATH_TIE_TOLERANCE)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ctl.encoder_path_check(self.own[:-1], self.reference)
        with self.assertRaises(ValueError):
            ctl.encoder_path_check(self.own[:1], self.reference[:1])
        bad = self.own.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ctl.encoder_path_check(bad, self.reference)


class OutputGuardTests(unittest.TestCase):
    def test_output_directory_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ctl.check_output_dir(root / 'not_created_yet')
            ctl.check_output_dir(root)
            (root / 'slurm-1.out').write_text('log')
            with self.assertRaisesRegex(ValueError, 'logs outside'):
                ctl.check_output_dir(root)
            (root / 'control_contract.json').write_text('{}')
            ctl.check_output_dir(root)
            (root / 'metrics.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                ctl.check_output_dir(root)
            with self.assertRaisesRegex(ValueError, 'official V12 evaluation root'):
                ctl.check_output_dir(root / 'subcellular_v12_evaluations_20260930' / 'neighbor_controls')
            self.assertEqual(sorted(item.name for item in root.iterdir()), ['control_contract.json', 'metrics.json', 'slurm-1.out'])

    def test_contract_differences_name_the_changed_keys(self):
        saved = dict(batch_size=8, reference_baseline_cache='/a', seed=42, old_only=1)
        current = dict(batch_size=16, reference_baseline_cache='/a', seed=42, new_only=2)
        self.assertEqual(ctl.contract_differences(saved, current), ['batch_size', 'new_only', 'old_only'])
        self.assertEqual(ctl.contract_differences(saved, dict(saved)), [])


class ReferenceCacheTests(unittest.TestCase):
    backbone = dict(backbone_kind='scgpt', source_sha256='src', weights_sha256='w', args_sha256='a', vocab_sha256='v')
    sources = {name: f'sha-{index}' for index, name in enumerate(ctl.REFERENCE_CACHE_SOURCES)}
    expected = dict(sample_name='S1', obs_sha256='o', genes_sha256='g', count_sizes={'nuclear.npy': 10, 'cytoplasmic.npy': 10},
        selected_sha256='s', backbone=backbone, sources=sources, max_tokens=512, slice_seed=11, cells=3)

    def contract(self, **changes):
        value = dict(schema_version=2, sample='/somewhere/else/S1', selected_sha256='s', max_tokens=512, seed=42, slice_seed=11,
            batch_size=8, torch_version='2.13', backbone=dict(self.backbone, weights='/other/path/model.pt', source='/other/model.py'),
            sources=dict(self.sources), **ctl.REFERENCE_CACHE_LITERALS,
            inputs=dict(obs_sha256='o', genes_sha256='g', counts={'nuclear.npy': dict(size=10, mtime_ns=1),
                                                                 'cytoplasmic.npy': dict(size=10, mtime_ns=2)}))
        value.update(changes)
        return value

    def test_pinned_literals_are_the_reference_evaluator_ones(self):
        text = Path(ctl.ev12.__file__).read_text()
        for key, value in ctl.REFERENCE_CACHE_LITERALS.items():
            self.assertIn(f"{key}='{value}'", text)
        for name in ctl.REFERENCE_CACHE_SOURCES:
            self.assertIn(f"'{name}'", text)
            self.assertIn(name, ctl.SOURCES)
        self.assertIn('weights_sha256', ctl.BACKBONE_IDENTITY)

    def test_stale_source_or_protocol_entries_are_not_reused(self):
        stale = dict(self.sources); stale['scripts/evaluate_subcellular_v12.py'] = 'older evaluator'
        missing = dict(self.sources); missing.pop('src/spatial_pretrain/data/subcellular_v5.py')
        for changes in (dict(sources=stale), dict(sources=missing), dict(tokenization='randomized ties'),
                        dict(readout='L2 normalized CLS'), dict(dtype='float32'),
                        dict(backbone=dict(self.backbone, vocab_sha256='other')),
                        dict(backbone=dict(self.backbone, backbone_kind='spatial')),
                        dict(backbone={key: value for key, value in self.backbone.items() if key != 'args_sha256'})):
            self.assertFalse(ctl.reference_cache_matches(self.contract(**changes), self.expected))
        for key in ('sources', 'tokenization', 'readout', 'dtype', 'backbone'):
            contract = self.contract(); contract.pop(key)
            self.assertFalse(ctl.reference_cache_matches(contract, self.expected))

    def entry(self, root, name, contract, fill):
        (root / name).mkdir()
        for readout in ('scgpt_whole', 'scgpt_nucleus', 'scgpt_cytoplasm'):
            np.save(root / name / f'{readout}.npy', np.full((3, 4), fill, dtype=np.float32))
        (root / name / 'complete.json').write_text(json.dumps(dict(contract=contract)))

    def test_contract_identity(self):
        self.assertTrue(ctl.reference_cache_matches(self.contract(), self.expected))
        self.assertTrue(ctl.reference_cache_matches(self.contract(batch_size=16, torch_version='x'), self.expected))
        for changes in (dict(selected_sha256='other'), dict(slice_seed=12), dict(schema_version=1), dict(max_tokens=256),
                        dict(sample='/a/S2'), dict(backbone=dict(weights_sha256='finetuned')), dict(seed=7)):
            self.assertFalse(ctl.reference_cache_matches(self.contract(**changes), self.expected))
        self.assertFalse(ctl.reference_cache_matches({}, self.expected))

    def test_official_cache_is_only_read_and_batch_size_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.entry(root, 'a16', self.contract(batch_size=16), 16.)
            self.entry(root, 'b8', self.contract(batch_size=8), 8.)
            self.entry(root, 'c_other', self.contract(selected_sha256='other'), 1.)
            (root / 'broken').mkdir(); (root / 'broken' / 'complete.json').write_text('{')
            before = {str(path.relative_to(root)): path.stat().st_mtime_ns for path in root.rglob('*')}
            values, info = ctl.load_reference_baselines(root, self.expected, 8, 4)
            self.assertEqual(set(values), set(ctl.ANCHOR_READOUTS))
            self.assertTrue((values['scgpt_whole'] == 8.).all())
            self.assertTrue(info['read_only_official_cache'] and info['reused'])
            self.assertEqual(info['batch_size'], 8)
            self.assertTrue(info['batch_size_matches'])
            self.assertFalse(info['torch_version_matches'])
            # Every matching entry is reported, in selection order.
            self.assertEqual([Path(item['path']).name for item in info['matching_entries']], ['b8', 'a16'])
            json.dumps(info, allow_nan=False)
            values, info = ctl.load_reference_baselines(root, self.expected, 32, 4)
            self.assertTrue((values['scgpt_nucleus'] == 16.).all())
            self.assertEqual(ctl.load_reference_baselines(root, dict(self.expected, selected_sha256='absent'), 8, 4), (None, None))
            self.assertEqual(ctl.load_reference_baselines(root, self.expected, 8, 512), (None, None))
            self.assertEqual(ctl.load_reference_baselines(root / 'missing', self.expected, 8, 4), (None, None))
            self.assertEqual(ctl.load_reference_baselines(None, self.expected, 8, 4), (None, None))
            after = {str(path.relative_to(root)): path.stat().st_mtime_ns for path in root.rglob('*')}
            self.assertEqual(before, after)

    def test_torch_version_ranks_entries_of_equal_batch_size_and_stale_sources_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.entry(root, 'a', self.contract(torch_version='2.12'), 1.)
            self.entry(root, 'b', self.contract(torch_version='2.13'), 2.)
            stale = dict(self.sources); stale['scripts/evaluate_subcellular_v12.py'] = 'older evaluator'
            self.entry(root, '0_stale', self.contract(sources=stale), 3.)
            values, info = ctl.load_reference_baselines(root, self.expected, 8, 4, '2.13')
            self.assertTrue((values['scgpt_whole'] == 2.).all())
            self.assertTrue(info['torch_version_matches'] and info['batch_size_matches'])
            self.assertEqual([Path(item['path']).name for item in info['matching_entries']], ['b', 'a'])


if __name__ == '__main__':
    unittest.main()
