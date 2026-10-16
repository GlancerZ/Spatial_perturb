"""Pure-logic tests of the V12 trivial count baselines; CPU only, synthetic arrays, no weights or real data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_subcellular_v12 as ev12
import evaluate_subcellular_v12_count_baselines as cb


def synthetic(seed=0, cells=80, queries=6, genes=10, train_cells=60):
    rng = np.random.default_rng(seed)
    gene = np.stack([rng.permutation(genes)[:queries] for _ in range(cells)])
    rate, fraction = rng.uniform(.002, .02, genes), rng.uniform(.2, .8, genes)
    depth = rng.integers(50, 400, cells)
    total = rng.poisson(rate[gene] * depth[:, None])
    nucleus = rng.binomial(total, fraction[gene])
    depth_nucleus = rng.binomial(depth, .4)
    data = dict(train=np.arange(cells) < train_cells, gene=gene, genes=genes, valid=np.ones(gene.shape, dtype=bool),
        nucleus=nucleus.astype(np.int64), cytoplasm=(total - nucleus).astype(np.int64),
        depth_nucleus=depth_nucleus.astype(np.int64), depth_cytoplasm=(depth - depth_nucleus).astype(np.int64),
        visible_genes=np.full(cells, 30), neighbor_counts=rng.poisson(rate[gene] * 800).astype(np.int64),
        neighbor_depth=np.full(cells, 800, dtype=np.int64))
    frame = pd.DataFrame(dict(center_row=np.arange(cells) * 3, full_cell_id=[f'cell{index}' for index in range(cells)],
        patient=[f'd{index // 10}' for index in range(cells)], split=np.where(data['train'], 'train', 'val')))
    available = {mode: np.ones(cells, dtype=bool) for mode in ev12.COUNT_MODES}
    available['nucleus'] = np.arange(cells) % 4 != 0
    return data, frame, available


def scoring_fixture():
    nucleus = np.array([[1, 0, 2], [0, 0, 3]])
    cytoplasm = np.array([[1, 0, 0], [0, 2, 1]])
    valid = np.array([[True, True, False], [True, True, True]])
    available = {mode: np.array([True, True]) for mode in ev12.COUNT_MODES}
    available['nucleus'] = np.array([True, False])
    parameters = {key: np.full(nucleus.shape, value) for key, value in
                  [('mu_total', .5), ('theta_total', 1.), ('p_nucleus', .25), ('kappa_allocation', 2.)]}
    return nucleus, cytoplasm, valid, available, parameters


def reference_nll(nucleus, cytoplasm, mu, theta, p, kappa):
    total = nucleus + cytoplasm
    result = -stats.nbinom.logpmf(total, theta, theta / (theta + mu))
    if total > 0:
        result -= stats.betabinom.logpmf(nucleus, total, p * kappa, (1 - p) * kappa)
    return result


def reference_calibrated(total, p, kappa):
    count = np.arange(total + 1)
    return float((stats.betabinom.pmf(count, total, p * kappa, (1 - p) * kappa) * np.abs(p - count / total)).sum())


class CohortAndOutputTests(unittest.TestCase):
    def test_cohort_rows_are_the_evaluator_expression(self):
        for cells in (37, 400, 12345):
            expected = np.sort(np.random.default_rng(42).choice(cells, min(400, cells), replace=False))
            rows = cb.select_cohort_rows(cells)
            np.testing.assert_array_equal(rows, expected)
            self.assertEqual(len(np.unique(rows)), min(400, cells))
        np.testing.assert_array_equal(cb.select_cohort_rows(37), np.arange(37))

    def test_completed_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'result'
            self.assertEqual(cb.prepare_output(directory), directory)
            (directory / 'anchor_mismatch.json').write_text('{}')
            cb.prepare_output(directory)
            (directory / 'metrics.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                cb.prepare_output(directory)

    def test_cohort_check_is_the_canonical_train10_val3_protocol(self):
        cells = pd.DataFrame(dict(full_cell_id=[f'c{index}' for index in range(5200)],
                                  patient=[f'd{index // 400:02d}' for index in range(5200)]))
        cells['split'] = np.where(np.arange(5200) < 4000, 'train', 'val')
        expected = ev12.ids_hash(cells.full_cell_id)
        check = cb.cohort_check(cells, expected)
        self.assertEqual((check['name'], check['status'], check['cells'], check['cohort_sha256']),
                         ('canonical_cohort', 'passed', 5200, expected))
        self.assertEqual((len(check['train_donors']), check['val_donors']), (10, ['d10', 'd11', 'd12']))
        self.assertEqual(cb.cohort_check(cells, ev12.CANONICAL_COHORT)['status'], 'failed')
        shorter = cells.iloc[:-1]
        self.assertEqual(cb.cohort_check(shorter, ev12.ids_hash(shorter.full_cell_id))['status'], 'failed')
        merged = cells.assign(patient=np.where(cells.patient.eq('d12'), 'd11', cells.patient))
        self.assertEqual(cb.cohort_check(merged, expected)['status'], 'failed')          # only two validation donors

    def test_anchor_failure_writes_the_diagnostic_then_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(SystemExit) as raised:
                cb.anchor_failure(directory, ['canonical_cohort'], dict(value=np.float64('nan'), cells=np.int64(3)))
            self.assertEqual(raised.exception.code, 2)
            saved = json.loads((directory / 'anchor_mismatch.json').read_text())
            self.assertEqual((saved['status'], saved['failed'], saved['value'], saved['cells']),
                             ('anchor_mismatch', ['canonical_cohort'], None, 3))
            self.assertFalse((directory / 'metrics.json').exists())

    def test_plain_makes_strict_json(self):
        value = cb.plain(dict(a=np.float64(1.5), b=np.arange(2), c=float('nan'), d=(1, np.int64(2)), e=Path('x')))
        self.assertEqual(value, dict(a=1.5, b=[0, 1], c=None, d=[1, 2], e='x'))


class VisibilityTests(unittest.TestCase):
    def test_visible_depth_excludes_valid_query_counts_only(self):
        depth = cb.visible_depth(np.array([10, 7]), np.array([[3, 4], [2, 5]]), np.array([[True, False], [True, True]]))
        np.testing.assert_array_equal(depth, [7, 0])
        with self.assertRaisesRegex(ValueError, 'exceed'):
            cb.visible_depth(np.array([4]), np.array([[3, 4]]), np.array([[True, True]]))

    def test_center_statistics_guard_detects_one_leaked_count(self):
        statistics = np.log1p(np.array([[10., 2000., 3., 4.]])).astype(np.float32)
        cb.check_center_statistics(statistics, np.array([10]), np.array([2000]))
        with self.assertRaisesRegex(ValueError, 'leak'):
            cb.check_center_statistics(statistics, np.array([11]), np.array([2000]))
        with self.assertRaisesRegex(ValueError, 'leak'):
            cb.check_center_statistics(statistics, np.array([10]), np.array([2001]))

    def availability_batch(self):
        pad, cls = 0, 1
        genes = torch.tensor([cls, 5, pad, pad])
        valid = torch.tensor([True, True, False, False])
        batch = dict(neighbor_present=torch.tensor([[True, True, False], [True, True, True], [False, False, False]]),
            neighbor_cell_identity=torch.tensor([[[0, 5], [0, 5], [-1, -1]], [[0, 1], [0, 7], [0, 8]], [[-1, -1]] * 3]),
            cell_identity=torch.tensor([[0, 0], [0, 1], [0, 2]]), relative_coords=torch.zeros(3, 3, 2),
            neighbor_gene_ids=genes.repeat(3, 3, 1), neighbor_valid=valid.repeat(3, 3, 1),
            nucleus_gene_ids=genes.repeat(3, 1), nucleus_valid=valid.repeat(3, 1),
            cytoplasm_gene_ids=genes.repeat(3, 1), cytoplasm_valid=valid.repeat(3, 1),
            nucleus_present=torch.tensor([True, False, True]), cytoplasm_present=torch.tensor([True, True, False]))
        batch['relative_coords'][1, 1, 0] = 200.           # beyond the 100 um radius
        batch['neighbor_valid'][1, 2, 1] = False            # CLS-only neighbour
        batch['nucleus_valid'][2, 1] = False                # flagged present but no gene token
        return batch, pad, cls

    def test_input_only_mode_availability(self):
        batch, pad, cls = self.availability_batch()
        available, present = cb.mode_available_from_batch(batch, pad, cls, 100.)
        np.testing.assert_array_equal(present, [[True, False, False], [False, False, False], [False, False, False]])
        expected = dict(nucleus=[True, False, False], cytoplasm=[True, True, False], local=[True, True, False],
                        spatial=[True, False, False], fused=[True, True, False])
        self.assertEqual(set(available), set(ev12.COUNT_MODES))
        for mode, value in expected.items():
            np.testing.assert_array_equal(available[mode], value, err_msg=mode)
        unlimited, _ = cb.mode_available_from_batch(batch, pad, cls, None)
        np.testing.assert_array_equal(unlimited['spatial'], [True, True, False])
        self.assertTrue(batch['neighbor_present'][0, 1])     # inputs are not modified in place

    def test_replica_equals_real_model_method(self):
        batch, pad, cls = self.availability_batch()
        official = cb.model_method_availability(batch, pad, cls, 100.)
        if official is None:
            self.skipTest('torch refuses a weightless SubcellularV12 instance')
        available, _ = cb.mode_available_from_batch(batch, pad, cls, 100.)
        for mode in ev12.COUNT_MODES:
            np.testing.assert_array_equal(official[mode], available[mode], err_msg=mode)


class EstimatorTests(unittest.TestCase):
    def test_gene_total_and_allocation_estimators(self):
        gene = np.array([0, 0, 1])
        statistics = cb.gene_statistics(gene, np.array([2., 0., 1.]), np.array([2., 0., 4.]), np.array([10., 30., 20.]), 3)
        totals = cb.total_estimators(statistics)
        self.assertAlmostEqual(totals['global_mean'], 2.)
        self.assertAlmostEqual(totals['global_rate'], .1)
        np.testing.assert_allclose(totals['gene_mean'], [1., 4., 2.])
        np.testing.assert_allclose(totals['gene_rate'], [.05, .2, .1])
        allocation = cb.allocation_estimators(statistics, 2.)
        self.assertAlmostEqual(allocation['global_fraction'], .5)
        np.testing.assert_allclose(allocation['gene_fraction'], [.75, 1 / 3, .5])
        np.testing.assert_allclose(cb.allocation_estimators(statistics, 1e9)['gene_fraction'], .5, atol=1e-6)
        with self.assertRaises(ValueError):
            cb.allocation_estimators(statistics, 0.)

    def test_gene_median_fraction_with_global_fallback(self):
        median = cb.gene_median_fraction(np.array([0, 0, 0, 1, 1]), np.array([0., 1., 1., 1., 0.]),
                                         np.array([1., 1., 1., 2., 0.]), 3)
        np.testing.assert_allclose(median, [1., .5, .75])

    def test_cell_effect_and_shrinkage(self):
        self.assertAlmostEqual(float(cb.shrunk_fraction(3, 10, .5, 10.)), .4)
        self.assertAlmostEqual(float(cb.shrunk_fraction(0, 0, .3, 5.)), .3)
        self.assertAlmostEqual(float(cb.gene_cell_allocation(.3, .5, .5)), .3)
        self.assertAlmostEqual(float(cb.gene_cell_allocation(.5, .8, .5)), .8)
        self.assertAlmostEqual(float(cb.gene_cell_allocation(.2, .6, .4)), .36)
        self.assertTrue(0 < float(cb.gene_cell_allocation(0., 0., 1.)) < 1e-6)

    def test_neighbor_total_and_blend_weight(self):
        value = cb.neighbor_total(np.array([2, 0]), np.array([100, 0]), np.array([50., 50.]), 2., np.array([9., 9.]))
        np.testing.assert_allclose(value, [2., 9.])
        rng = np.random.default_rng(1)
        truth = rng.uniform(.1, 3., 5000)
        total = rng.poisson(truth).astype(np.float64)
        constant = np.full(5000, truth.mean())
        self.assertEqual(cb.choose_blend_weight(total, constant, truth, (0., .5, 1.))[0], 1.)
        self.assertEqual(cb.choose_blend_weight(total, truth, constant, (0., .5, 1.))[0], 0.)

    def test_selection_losses(self):
        self.assertAlmostEqual(float(cb.binomial_nll(1., 2., .5)), 2 * np.log(2))
        self.assertAlmostEqual(float(cb.binomial_nll(0., 0., .3)), 0.)
        self.assertAlmostEqual(float(cb.poisson_nll(2., 3.)), 3 - 2 * np.log(3))

    def test_train_observations_exclude_validation_and_padding(self):
        data, _, _ = synthetic()
        data['valid'][0, 0] = False
        obs = cb.train_observations(data)
        self.assertEqual(len(obs['cell']), 60 * 6 - 1)
        self.assertTrue(data['train'][obs['cell']].all())
        self.assertFalse(obs['mask'][60:].any())
        np.testing.assert_array_equal(obs['total'], (data['nucleus'] + data['cytoplasm'])[obs['mask']])

    def test_fitted_parameters_ignore_validation_donors(self):
        data, _, _ = synthetic()
        first = cb.fit_baselines(data)
        totals, allocations = cb.predict_components(data, first)
        theta, kappa, _ = cb.fit_dispersions(data, totals, allocations, min_informative=5)
        changed = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in data.items()}
        for key in ('nucleus', 'cytoplasm', 'depth_nucleus', 'depth_cytoplasm', 'neighbor_counts', 'neighbor_depth'):
            changed[key][60:] = changed[key][60:] * 7 + 3
        second = cb.fit_baselines(changed)
        for key, value in first.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, second[key], err_msg=key)
            else:
                self.assertEqual(value, second[key], key)
        totals2, allocations2 = cb.predict_components(changed, second)
        theta2, kappa2, _ = cb.fit_dispersions(changed, totals2, allocations2, min_informative=5)
        for name in cb.TOTALS:
            np.testing.assert_array_equal(theta[name], theta2[name], err_msg=name)
            np.testing.assert_array_equal(totals[name][:60], totals2[name][:60], err_msg=name)
        for name in cb.ALLOCATIONS:
            np.testing.assert_array_equal(kappa[name], kappa2[name], err_msg=name)
            np.testing.assert_array_equal(allocations[name][:60], allocations2[name][:60], err_msg=name)

    def test_predictions_never_read_targets_and_use_visible_depth(self):
        data, _, _ = synthetic()
        fitted = cb.fit_baselines(data)
        totals, allocations = cb.predict_components(data, fitted)
        self.assertEqual(set(totals), set(cb.TOTALS))
        self.assertEqual(set(allocations), set(cb.ALLOCATIONS))
        hidden = dict(data, nucleus=data['nucleus'] * 0 + 99, cytoplasm=data['cytoplasm'] * 0 + 99)
        totals2, allocations2 = cb.predict_components(hidden, fitted)
        for name in cb.TOTALS:
            np.testing.assert_array_equal(totals[name], totals2[name])
        for name in cb.ALLOCATIONS:
            np.testing.assert_array_equal(allocations[name], allocations2[name])
        deeper = dict(data, depth_nucleus=data['depth_nucleus'] * 2, depth_cytoplasm=data['depth_cytoplasm'] * 2)
        np.testing.assert_allclose(cb.predict_components(deeper, fitted)[0]['t_gene_depth'], 2 * totals['t_gene_depth'])
        np.testing.assert_allclose(totals['t_gene_depth'],
            fitted['gene_rate'][data['gene']] * (data['depth_nucleus'] + data['depth_cytoplasm'])[:, None])
        np.testing.assert_allclose(totals['t_zero'], cb.BOUNDS['mu_total'][0])
        np.testing.assert_allclose(allocations['a_global'], fitted['global_fraction'])
        for value in allocations.values():
            self.assertTrue(((value >= 1e-4) & (value <= 1 - 1e-4)).all())

    def test_parameter_clipping_and_padding_fill(self):
        valid = np.array([[True, False]])
        result = cb.clip_parameters(np.array([[0., 5.]]), np.array([[1e9, 5.]]), np.array([[1., .2]]), np.array([[0., 5.]]), valid)
        self.assertEqual(result['mu_total'].tolist(), [[1e-4, 1.]])
        self.assertEqual(result['theta_total'].tolist(), [[1e5, 1.]])
        self.assertEqual(result['p_nucleus'].tolist(), [[1 - 1e-4, .5]])
        self.assertEqual(result['kappa_allocation'].tolist(), [[2e-3, 1.]])


class LikelihoodTests(unittest.TestCase):
    def test_joint_and_parts_match_scipy(self):
        n = torch.tensor([0., 1., 2., 7.], dtype=torch.float64)
        c = torch.tensor([0., 0., 3., 1.], dtype=torch.float64)
        mu, theta, p, kappa = [torch.full_like(n, value) for value in (1.7, .8, .3, 4.)]
        joint = cb.joint_nll(n, c, mu, theta, p, kappa)
        expected = [reference_nll(int(a), int(b), 1.7, .8, .3, 4.) for a, b in zip(n, c)]
        np.testing.assert_allclose(joint.numpy(), expected, rtol=1e-10)
        official = ev12.joint_count_nll(n, c, dict(mu_total=mu, theta_total=theta, p_nucleus=p, kappa_allocation=kappa))
        np.testing.assert_allclose(official.numpy(), expected, rtol=1e-10)
        total_part = cb.nb_nll(n + c, mu, theta)
        np.testing.assert_allclose(total_part.numpy(), -stats.nbinom.logpmf((n + c).numpy(), .8, .8 / 2.5), rtol=1e-10)
        self.assertAlmostEqual(float((joint - total_part)[0]), 0.)

    def test_grouped_log_grid_finds_each_group_optimum(self):
        group = np.array([0, 0, 1, 1, 1])
        target = torch.tensor([-1., -1., 2., 2., 2.], dtype=torch.float64)
        value, loss = cb.fit_grouped_log_scalar(lambda item: (item.log10() - target) ** 2, 1e-3, 1e5, group, 3)
        np.testing.assert_allclose(np.log10(value[:2]), [-1., 2.], atol=.01)
        self.assertLess(loss[:2].max(), 1e-3)
        self.assertTrue(1e-3 <= value[2] <= 1e5)             # empty group stays inside the bounds

    def test_theta_and_kappa_maximum_likelihood_recover_simulation(self):
        rng = np.random.default_rng(3)
        size = 20000
        total = torch.as_tensor(rng.negative_binomial(2., 2. / (2. + 3.), size).astype(np.float64))
        zeros, ones = torch.zeros_like(total), torch.ones_like(total)
        fit = cb.fit_dispersion(lambda item: cb.joint_nll(zeros, total, 3. * ones, item, .5 * ones, ones),
                                cb.BOUNDS['theta_total'], np.zeros(size, dtype=np.int64), 1, np.ones(size, dtype=bool), 20)
        self.assertTrue(1.7 < fit['global_value'] < 2.3, fit['global_value'])
        probability = rng.beta(.3 * 8, .7 * 8, size)
        nucleus = torch.as_tensor(rng.binomial(10, probability).astype(np.float64))
        cytoplasm = 10. - nucleus
        fit = cb.fit_dispersion(lambda item: cb.joint_nll(nucleus, cytoplasm, ones, ones, .3 * ones, item),
                                cb.BOUNDS['kappa_allocation'], np.zeros(size, dtype=np.int64), 1, np.ones(size, dtype=bool), 20)
        self.assertTrue(6.5 < fit['global_value'] < 9.5, fit['global_value'])

    def test_sparse_gene_falls_back_to_global_dispersion(self):
        rng = np.random.default_rng(4)
        total = torch.as_tensor(rng.poisson(2., 300).astype(np.float64))
        gene = np.r_[np.zeros(295, dtype=np.int64), np.ones(5, dtype=np.int64)]
        zeros, ones = torch.zeros_like(total), torch.ones_like(total)
        fit = cb.fit_dispersion(lambda item: cb.joint_nll(zeros, total, 2. * ones, item, .5 * ones, ones),
                                cb.BOUNDS['theta_total'], gene, 3, np.ones(300, dtype=bool), 20)
        self.assertEqual((fit['fitted_genes'], fit['fallback_genes']), (1, 2))
        self.assertEqual(fit['per_gene'][1], fit['global_value'])
        self.assertEqual(fit['per_gene'][2], fit['global_value'])

    def test_gene_free_estimators_keep_one_global_dispersion(self):
        cells = 40
        # Gene 0: T=2 always (underdispersed), N in {0,2} (extreme allocation). Gene 1: T in {0,8} (overdispersed), N=T/2.
        total = np.stack([np.full(cells, 2), np.where(np.arange(cells) < 10, 8, 0)], 1)
        nucleus = np.stack([np.where(np.arange(cells) % 2 == 0, 2, 0), total[:, 1] // 2], 1)
        data = dict(valid=np.ones((cells, 2), dtype=bool), train=np.ones(cells, dtype=bool), genes=2,
                    gene=np.tile([0, 1], (cells, 1)), nucleus=nucleus, cytoplasm=total - nucleus)
        mean, half = np.full((cells, 2), 2.), np.full((cells, 2), .5)
        theta, kappa, summary = cb.fit_dispersions(data, dict(t_gene_mean=mean, t_cell_mean=mean),
                                                   dict(a_gene=half, a_global=half), min_informative=5)
        self.assertGreater(theta['t_gene_mean'][0, 0], 100 * theta['t_gene_mean'][0, 1])
        self.assertGreater(kappa['a_gene'][0, 1], 100 * kappa['a_gene'][0, 0])
        for name, value in (('t_cell_mean', theta['t_cell_mean']), ('a_global', kappa['a_global'])):
            np.testing.assert_array_equal(value, np.full((cells, 2), summary[name]['global_value']), err_msg=name)
            self.assertEqual((summary[name]['scope'], summary[name]['fitted_genes'], summary[name]['fallback_genes']),
                             ('global', 0, 2))
        for name in ('t_gene_mean', 'a_gene'):
            self.assertEqual((summary[name]['scope'], summary[name]['fitted_genes']), ('per_gene_with_global_fallback', 2))
        self.assertEqual(summary['t_gene_mean']['global_value'], summary['t_cell_mean']['global_value'])
        self.assertEqual(set(cb.GENE_FREE), {'t_zero', 't_cell_mean', 'a_global', 'a_cell'})
        everything, _, _ = cb.fit_dispersions(data, dict(t_cell_mean=mean), {}, min_informative=5, gene_free=())
        self.assertGreater(len(np.unique(everything['t_cell_mean'])), 1)      # the leak the gene-free list prevents

    def test_expected_allocation_abs_error_is_the_exact_beta_binomial_mean(self):
        total = np.array([[0, 1, 1], [2, 5, 9]])
        p = np.array([[.3, .3, .9], [.25, .6, .1]])
        kappa = np.array([[4., 4., .5], [2., 30., 1e4]])
        value = cb.expected_allocation_abs_error(total, p, kappa).numpy()
        expected = np.zeros(total.shape)
        for index in np.ndindex(*total.shape):
            if total[index]:
                expected[index] = reference_calibrated(int(total[index]), p[index], kappa[index])
        np.testing.assert_allclose(value, expected, rtol=1e-7)
        self.assertEqual(value[0, 0], 0.)                                     # T=0 has no allocation
        np.testing.assert_allclose(value[0, 1:], [2 * .3 * .7, 2 * .9 * .1], rtol=1e-9)   # T=1: 2q(1-q) for any kappa
        self.assertTrue((value[1] > 0).all())                                 # a calibrated p is never error-free
        with self.assertRaises(ValueError):
            cb.expected_allocation_abs_error(total, p[:1], kappa)


class ScoringTests(unittest.TestCase):
    def test_per_cell_sums_follow_the_evaluator_definitions(self):
        nucleus, cytoplasm, valid, available, parameters = scoring_fixture()
        frame = cb.score_parameters(nucleus, cytoplasm, valid, available, parameters)
        self.assertEqual(len(frame), 2 * len(ev12.COUNT_MODES))
        local = frame[frame['mode'].eq('local')].set_index('cell')
        self.assertEqual(local.query_count.tolist(), [2, 3])
        self.assertEqual(local.positive_query_count.tolist(), [1, 2])
        np.testing.assert_allclose(local.total_abs_error_sum, [2., 5.5])
        np.testing.assert_allclose(local.nucleus_abs_error_sum, [1., 3.125])
        np.testing.assert_allclose(local.cytoplasm_abs_error_sum, [1., 2.625])
        np.testing.assert_allclose(local.allocation_abs_error_sum, [.25, .75])
        expected = [reference_nll(1, 1, .5, 1., .25, 2.) + reference_nll(0, 0, .5, 1., .25, 2.),
                    reference_nll(0, 0, .5, 1., .25, 2.) + reference_nll(0, 2, .5, 1., .25, 2.) + reference_nll(3, 1, .5, 1., .25, 2.)]
        np.testing.assert_allclose(local.nll_sum, expected, rtol=1e-10)
        np.testing.assert_allclose(local.nll_total_sum + local.nll_allocation_sum, local.nll_sum, rtol=1e-10)
        allocation = [-stats.betabinom.logpmf(1, 2, .5, 1.5),
                      -stats.betabinom.logpmf(0, 2, .5, 1.5) - stats.betabinom.logpmf(3, 4, .5, 1.5)]
        np.testing.assert_allclose(local.nll_allocation_sum, allocation, rtol=1e-9)
        np.testing.assert_allclose(local.allocation_calibrated_abs_error_sum, [reference_calibrated(2, .25, 2.),
            reference_calibrated(2, .25, 2.) + reference_calibrated(4, .25, 2.)], rtol=1e-9)
        missing = frame[frame['mode'].eq('nucleus')].set_index('cell').loc[1]
        self.assertEqual(float(missing[['query_count', 'positive_query_count', *cb.SUM_FIELDS, 'nll_total_sum',
                                        'allocation_calibrated_abs_error_sum']].astype(float).abs().sum()), 0.)

    def test_mode_specific_parameters_are_selected_by_mode(self):
        nucleus, cytoplasm, valid, available, parameters = scoring_fixture()
        by_mode = {mode: dict(parameters) for mode in ev12.COUNT_MODES}
        by_mode['spatial'] = dict(parameters, mu_total=np.full(nucleus.shape, 2.))
        frame = cb.score_parameters(nucleus, cytoplasm, valid, available, by_mode)
        spatial = frame[frame['mode'].eq('spatial')].set_index('cell')
        fused = frame[frame['mode'].eq('fused')].set_index('cell')
        np.testing.assert_allclose(spatial.total_abs_error_sum, [2., 4.])     # |2-2|+|2-0| ; |2-0|+|2-2|+|2-4|
        np.testing.assert_allclose(fused.total_abs_error_sum, [2., 5.5])

    def test_aggregation_averages_allocation_over_positive_queries_only(self):
        nucleus, cytoplasm, valid, available, parameters = scoring_fixture()
        cells = pd.DataFrame(dict(center_row=[11, 12], full_cell_id=['x', 'y'], patient=['a', 'b'], split=['val', 'val']))
        frame = cb.attach_cells(cb.score_parameters(nucleus, cytoplasm, valid, available, parameters), cells)
        rows = {cb.row_key(row): row for row in cb.summarize(frame, 'demo', dict(source='test'))}
        pooled = rows['val', 'local', 'pooled']
        self.assertEqual((pooled['query_count'], pooled['positive_query_count'], pooled['available_cells']), (5, 3, 2))
        self.assertAlmostEqual(pooled['total_abs_error'], 7.5 / 5)
        self.assertAlmostEqual(pooled['nucleus_abs_error'], 4.125 / 5)
        self.assertAlmostEqual(pooled['cytoplasm_abs_error'], 3.625 / 5)
        self.assertAlmostEqual(pooled['allocation_abs_error'], 1. / 3)
        self.assertAlmostEqual(pooled['nll_total_part'] + pooled['nll_allocation_part'], pooled['nll'])
        self.assertAlmostEqual(pooled['allocation_nll_per_positive'], pooled['nll_allocation_part'] * 5 / 3)
        self.assertAlmostEqual(pooled['allocation_abs_error_calibrated'],
                               (2 * reference_calibrated(2, .25, 2.) + reference_calibrated(4, .25, 2.)) / 3)
        self.assertIsNone(rows['val', 'nucleus', 'b']['allocation_abs_error_calibrated'])
        nucleus_row = rows['val', 'nucleus', 'pooled']
        self.assertEqual((nucleus_row['query_count'], nucleus_row['available_cells']), (2, 1))
        self.assertAlmostEqual(nucleus_row['allocation_abs_error'], .25)
        absent = rows['val', 'nucleus', 'b']
        self.assertEqual(absent['query_count'], 0)
        self.assertIsNone(absent['nll'])
        self.assertIsNone(absent['allocation_abs_error'])
        self.assertIsNone(absent['nll_total_part'])
        for field in ('split', 'mode', 'donor', *cb.COUNT_FIELDS, *cb.METRICS, *cb.PART_METRICS, *cb.DIAGNOSTICS, 'baseline'):
            self.assertIn(field, pooled)
        self.assertEqual(frame.center_row.tolist()[:2], [11, 12])

    def test_zero_total_queries_have_no_allocation_term(self):
        valid = np.ones((1, 2), dtype=bool)
        available = {mode: np.array([True]) for mode in ev12.COUNT_MODES}
        parameters = {key: np.full((1, 2), value) for key, value in
                      [('mu_total', .2), ('theta_total', 3.), ('p_nucleus', .9), ('kappa_allocation', 5.)]}
        frame = cb.score_parameters(np.zeros((1, 2)), np.zeros((1, 2)), valid, available, parameters)
        self.assertEqual(frame.positive_query_count.sum(), 0)
        self.assertEqual(frame.allocation_abs_error_sum.abs().sum(), 0.)
        self.assertEqual(frame.nll_allocation_sum.abs().sum(), 0.)
        np.testing.assert_allclose(frame.nll_sum, frame.nll_total_sum)

    def test_end_to_end_synthetic_baselines(self):
        data, cells, available = synthetic()
        outcome = cb.run_baselines(data, cells, available, min_informative=5)
        rows = outcome['rows']
        donors_per_split = dict(train=6, val=2)
        self.assertEqual(len(rows), len(cb.JOINT) * len(ev12.COUNT_MODES) * sum(count + 1 for count in donors_per_split.values()))
        self.assertEqual({row['baseline'] for row in rows}, {f'{t}+{a}' for t, a in cb.JOINT})
        index = {(row['baseline'], *cb.row_key(row)): row for row in rows}
        for row in rows:
            for metric in cb.METRICS:
                self.assertTrue(np.isfinite(row[metric]) and row[metric] >= 0, (row['baseline'], metric))
            self.assertAlmostEqual(row['nll_total_part'] + row['nll_allocation_part'], row['nll'], places=9)
            same = index[('t_gene_depth+a_gene', *cb.row_key(row))]
            self.assertEqual([row[field] for field in cb.COUNT_FIELDS], [same[field] for field in cb.COUNT_FIELDS])
        pooled = index['t_gene_depth+a_gene', 'val', 'local', 'pooled']
        self.assertEqual((pooled['query_count'], pooled['available_cells']), (20 * 6, 20))
        self.assertEqual(index['t_gene_depth+a_gene', 'val', 'nucleus', 'pooled']['available_cells'], 15)
        self.assertLess(pooled['nll'], index['t_zero+a_gene_median', 'val', 'local', 'pooled']['nll'])
        self.assertLess(pooled['nll_total_part'], index['t_cell_mean+a_cell', 'val', 'local', 'pooled']['nll_total_part'])
        total = (data['nucleus'] + data['cytoplasm'])[60:]
        self.assertAlmostEqual(index['t_zero+a_gene_median', 'val', 'local', 'pooled']['total_abs_error'],
                               float(np.abs(total - 1e-4).mean()))
        frame = outcome['frame']
        self.assertEqual(len(frame), len(cb.JOINT) * len(ev12.COUNT_MODES) * 80)
        self.assertFalse(frame.duplicated(['baseline', 'full_cell_id', 'mode']).any())
        summary = cb.target_summary(data)
        self.assertEqual(summary['val']['queries'], 120)
        self.assertAlmostEqual(summary['val']['mean_total'], float(total.mean()))


class ComparisonTests(unittest.TestCase):
    def rows(self, **changes):
        row = dict(baseline='b', split='val', mode='fused', donor='pooled', query_count=10, positive_query_count=4,
                   available_cells=2, nll=2., total_abs_error=.5, nucleus_abs_error=.2, cytoplasm_abs_error=.3,
                   allocation_abs_error=.4)
        row.update(changes)
        return [row]

    def test_skill_definition(self):
        self.assertAlmostEqual(cb.skill(1., 2.), .5)
        self.assertAlmostEqual(cb.skill(3., 2.), -.5)
        self.assertIsNone(cb.skill(None, 2.))
        self.assertIsNone(cb.skill(1., None))
        self.assertIsNone(cb.skill(1., 0.))

    def test_side_by_side_table(self):
        model = self.rows(nll=1., total_abs_error=.6, allocation_abs_error=None)
        table = cb.side_by_side(self.rows(), model + self.rows(mode='local'), 'ref')
        self.assertEqual(len(table), 1)
        item = table[0]
        self.assertEqual((item['reference'], item['baseline'], item['mode']), ('ref', 'b', 'fused'))
        self.assertAlmostEqual(item['nll_skill'], .5)
        self.assertAlmostEqual(item['total_abs_error_skill'], -.2)
        self.assertIsNone(item['allocation_abs_error_skill'])
        self.assertEqual((item['nll_baseline'], item['nll_model']), (2., 1.))
        self.assertNotIn('nll_total_part_skill', item)
        self.assertIsNone(item['baseline_inputs_within_model_mode'])          # estimator names unknown
        baseline = self.rows(mode='spatial', total_estimator='t_gene_depth', allocation_estimator='a_gene',
                             allocation_abs_error_calibrated=.3)
        item = cb.side_by_side(baseline, self.rows(mode='spatial'), 'ref')[0]
        self.assertIs(item['baseline_inputs_within_model_mode'], False)       # centre depth is invisible to spatial
        self.assertEqual(item['allocation_abs_error_calibrated_baseline'], .3)
        self.assertNotIn('allocation_abs_error_calibrated_model', item)
        self.assertNotIn('allocation_abs_error_calibrated_skill', item)
        item = cb.side_by_side(baseline, self.rows(mode='spatial', allocation_abs_error_calibrated=.2), 'ref')[0]
        self.assertEqual(item['allocation_abs_error_calibrated_model'], .2)

    def test_like_for_like_input_flags(self):
        self.assertEqual(set(cb.ESTIMATOR_INPUTS), set(cb.TOTALS) | set(cb.ALLOCATIONS))
        self.assertEqual(set(cb.MODE_INPUTS), set(ev12.COUNT_MODES))
        for total, allocation in cb.JOINT:
            self.assertIs(cb.inputs_within_mode(total, allocation, 'fused'), True)
        self.assertIs(cb.inputs_within_mode('t_gene_depth', 'a_gene_cell', 'local'), True)
        self.assertIs(cb.inputs_within_mode('t_gene_depth_nbr', 'a_gene', 'local'), False)
        for mode in ('nucleus', 'cytoplasm', 'spatial'):
            self.assertIs(cb.inputs_within_mode('t_gene_depth', 'a_gene', mode), False)
            self.assertIs(cb.inputs_within_mode('t_gene_mean', 'a_gene_cell', mode), False)
            self.assertIs(cb.inputs_within_mode('t_gene_mean', 'a_gene', mode), True)
            self.assertIs(cb.inputs_within_mode('t_zero', 'a_gene_median', mode), True)
        self.assertIsNone(cb.inputs_within_mode('v12_decoder', 'v12_decoder', 'fused'))
        self.assertIsNone(cb.inputs_within_mode('t_gene_mean', 'a_gene', 'concat'))

    def test_scorer_anchor_flag_needs_a_numeric_cross_check(self):
        checks = [dict(name='reference_query_counts:x', status='passed'),
                  dict(name='reference_file_expected_values:x', status='passed'),
                  dict(name='scorer_vs_evaluator_score_masked_counts', status='skipped')]
        self.assertFalse(cb.scorer_is_anchored(checks))
        self.assertFalse(cb.scorer_is_anchored(checks + [dict(name='rescored_model_vs_reference:x', status='failed')]))
        self.assertTrue(cb.scorer_is_anchored(checks + [dict(name='rescored_model_vs_reference:x', status='passed')]))
        self.assertTrue(cb.scorer_is_anchored([dict(name='scorer_vs_evaluator_score_masked_counts', status='passed')]))

    def test_query_count_anchor(self):
        self.assertEqual(cb.count_mismatches(self.rows(), self.rows(nll=9.)), [])
        problems = cb.count_mismatches(self.rows(), self.rows(positive_query_count=5))
        self.assertEqual(len(problems), 1)
        self.assertEqual((problems[0]['field'], problems[0]['baseline'], problems[0]['reference']), ('positive_query_count', 4, 5))
        problems = cb.count_mismatches(self.rows(), self.rows() + self.rows(mode='local'))
        self.assertEqual([problem['reason'] for problem in problems], ['missing in baselines'])
        problems = cb.count_mismatches(self.rows() + self.rows(donor='x'), self.rows())
        self.assertEqual([problem['reason'] for problem in problems], ['missing in reference'])

    def test_per_cell_anchor(self):
        frame = pd.DataFrame(dict(full_cell_id=['a', 'a', 'b'], mode=['fused', 'local', 'fused'],
                                  query_count=[64, 64, 0], positive_query_count=[7, 7, 0]))
        self.assertEqual(cb.per_cell_mismatches(frame, frame.copy())[0], 0)
        changed = frame.copy(); changed.loc[1, 'positive_query_count'] = 8
        self.assertEqual(cb.per_cell_mismatches(frame, changed)[0], 1)
        self.assertEqual(cb.per_cell_mismatches(frame, frame.iloc[:2])[0], 1)

    def test_expected_value_and_metric_difference_anchors(self):
        good = self.rows(nll=.6010, total_abs_error=.3861, allocation_abs_error=.3899)
        self.assertTrue(cb.expected_value_check(good, 3400)['passed'])
        self.assertFalse(cb.expected_value_check(self.rows(nll=.61, total_abs_error=.386, allocation_abs_error=.39), 3400)['passed'])
        self.assertFalse(cb.expected_value_check(self.rows(mode='local'), 3400)['passed'])
        self.assertIsNone(cb.expected_value_check(good, 100))
        worst, compared = cb.metric_differences(self.rows(), self.rows(nll=2.001, allocation_abs_error=None))
        self.assertEqual(compared, 1)
        self.assertAlmostEqual(worst['nll'], .001)
        self.assertEqual(worst['allocation_abs_error'], float('inf'))
        self.assertEqual(worst['total_abs_error'], 0.)


if __name__ == '__main__':
    unittest.main()
