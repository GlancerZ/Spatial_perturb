"""V12 inference-ablation logic; CPU-only synthetic tensors, compute nodes only."""
from pathlib import Path
import csv
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_subcellular_v12 as ev12
import evaluate_subcellular_v12_ablations as ab

BATCH_LEVEL = ('none', 'drop_neighbors', 'lr_off', 'lr_shuffled', 'swap_compartments')


def toy_batch(seed=0):
    rng = np.random.default_rng(seed)
    size, k = 3, 4

    def tokens(*shape):
        return torch.from_numpy(rng.integers(2, 40, shape))

    def values(*shape):
        return torch.from_numpy(rng.random(shape).astype(np.float32))

    nucleus, cytoplasm = tokens(size, 2), tokens(size, 2)
    return dict(
        nucleus_gene_ids=tokens(size, 5), nucleus_bins=values(size, 5),
        nucleus_valid=torch.ones(size, 5, dtype=torch.bool), nucleus_log_counts=values(size, 5),
        nucleus_present=torch.tensor([True, True, False]),
        cytoplasm_gene_ids=tokens(size, 3), cytoplasm_bins=values(size, 3),
        cytoplasm_valid=torch.ones(size, 3, dtype=torch.bool), cytoplasm_log_counts=values(size, 3),
        cytoplasm_present=torch.tensor([True, False, True]),
        neighbor_gene_ids=tokens(size, k, 6), neighbor_bins=values(size, k, 6),
        neighbor_valid=torch.ones(size, k, 6, dtype=torch.bool),
        neighbor_present=torch.tensor([[True, True, True, False], [True, False, True, True], [False, True, False, False]]),
        relative_coords=values(size, k, 2), cell_identity=torch.tensor([[2, 10], [2, 11], [2, 12]]),
        neighbor_cell_identity=torch.from_numpy(rng.integers(100, 200, (size, k, 2))),
        center_statistics=values(size, 4), neighbor_lr_bias=values(size, k, k), neighbor_lr_coverage=values(size, k, k),
        center_lr_bias=values(size, k), center_lr_coverage=values(size, k),
        query_gene_ids=tokens(size, 2), query_nucleus_counts=nucleus, query_cytoplasm_counts=cytoplasm,
        query_total_counts=nucleus + cytoplasm, query_valid_mask=torch.ones(size, 2, dtype=torch.bool))


def clone(batch):
    return {key: value.clone() for key, value in batch.items()}


def changed_keys(first, second):
    assert set(first) == set(second)
    return {key for key in first if first[key].shape != second[key].shape or not torch.equal(first[key], second[key])}


class BatchPerturbationTests(unittest.TestCase):
    def test_none_is_identity_in_a_new_container(self):
        batch = toy_batch()
        result = ab.apply_ablation('none', batch)
        self.assertIsNot(result, batch)
        self.assertEqual(changed_keys(batch, result), set())
        for name in ('shuffle_neighbors', 'whole_as_nucleus'):
            # Data-level ablations arrive already encoded; no tensor-level change is added.
            self.assertEqual(changed_keys(batch, ab.apply_ablation(name, batch)), set())

    def test_transforms_never_mutate_inputs_or_touch_queries_and_targets(self):
        for name in BATCH_LEVEL:
            batch = toy_batch()
            snapshot = clone(batch)
            result = ab.apply_ablation(name, batch, 7)
            self.assertEqual(changed_keys(batch, snapshot), set(), name)
            self.assertEqual(set(result), set(batch), name)
            for key in batch:
                if key.startswith('query_'):
                    self.assertIs(result[key], batch[key], f'{name} {key}')

    def test_swap_exchanges_every_compartment_tensor_and_is_an_involution(self):
        batch = toy_batch()
        swapped = ab.swap_compartments(batch)
        for field in ab.COMPARTMENT_FIELDS:
            self.assertTrue(torch.equal(swapped[f'nucleus_{field}'], batch[f'cytoplasm_{field}']))
            self.assertTrue(torch.equal(swapped[f'cytoplasm_{field}'], batch[f'nucleus_{field}']))
        torch.testing.assert_close(swapped['center_statistics'], batch['center_statistics'][:, [1, 0, 3, 2]])
        expected = {f'{name}_{field}' for name in ('nucleus', 'cytoplasm') for field in ab.COMPARTMENT_FIELDS}
        self.assertEqual(changed_keys(batch, swapped), expected | {'center_statistics'})
        self.assertEqual(changed_keys(batch, ab.swap_compartments(swapped)), set())

    def test_swap_refuses_unknown_compartment_tensors(self):
        batch = toy_batch()
        batch['nucleus_extra'] = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, 'per-compartment'):
            ab.swap_compartments(batch)
        batch = toy_batch()
        del batch['cytoplasm_log_counts']
        with self.assertRaisesRegex(ValueError, 'per-compartment'):
            ab.swap_compartments(batch)

    def test_drop_marks_all_neighbors_absent_and_nothing_else(self):
        batch = toy_batch()
        dropped = ab.drop_neighbors(batch)
        self.assertEqual(int(dropped['neighbor_present'].sum()), 0)
        self.assertEqual(dropped['neighbor_present'].dtype, torch.bool)
        self.assertEqual(changed_keys(batch, dropped), {'neighbor_present'})

    def test_lr_off_zeroes_exactly_the_bias_tensors(self):
        batch = toy_batch()
        result = ab.lr_off(batch)
        self.assertEqual(changed_keys(batch, result), set(ab.LR_BIAS_KEYS))
        for key in ab.LR_BIAS_KEYS:
            self.assertEqual(float(result[key].abs().sum()), 0.)
            self.assertEqual(result[key].dtype, batch[key].dtype)

    def test_random_cycle_is_one_cycle_without_fixed_points(self):
        rng = np.random.default_rng(3)
        self.assertEqual(len(ab.random_cycle(0, rng)), 0)
        np.testing.assert_array_equal(ab.random_cycle(1, rng), [0])
        for count in (2, 3, 7, 12):
            source = ab.random_cycle(count, rng)
            np.testing.assert_array_equal(np.sort(source), np.arange(count))
            self.assertFalse((source == np.arange(count)).any())
            position, visited = 0, set()
            while position not in visited:
                visited.add(position)
                position = int(source[position])
            self.assertEqual(len(visited), count)

    def test_lr_shuffle_preserves_values_and_moves_every_present_entry(self):
        batch = toy_batch()
        result = ab.lr_shuffled(batch, 42)
        self.assertEqual(changed_keys(batch, result), set(ab.LR_BIAS_KEYS))
        present = batch['neighbor_present'].numpy()
        for index, row in enumerate(present):
            slots = np.flatnonzero(row)
            old, new = batch['center_lr_bias'][index].numpy(), result['center_lr_bias'][index].numpy()
            np.testing.assert_array_equal(np.sort(old[slots]), np.sort(new[slots]))
            np.testing.assert_array_equal(old[~row], new[~row])
            pair = row[:, None] & row[None, :] & ~np.eye(len(row), dtype=bool)
            old, new = batch['neighbor_lr_bias'][index].numpy(), result['neighbor_lr_bias'][index].numpy()
            np.testing.assert_array_equal(np.sort(old[pair]), np.sort(new[pair]))
            np.testing.assert_array_equal(old[~pair], new[~pair])
            if len(slots) >= 2:
                self.assertTrue((old[pair] != new[pair]).all())
                self.assertTrue((batch['center_lr_bias'][index].numpy()[slots] != result['center_lr_bias'][index].numpy()[slots]).all())
        # One present neighbor: nothing can be reassigned.
        self.assertTrue(torch.equal(batch['center_lr_bias'][2], result['center_lr_bias'][2]))

    def test_lr_shuffle_is_deterministic_seeded_and_batch_independent(self):
        batch = toy_batch()
        first, second = ab.lr_shuffled(batch, 42), ab.lr_shuffled(clone(batch), 42)
        self.assertEqual(changed_keys(first, second), set())
        other = ab.lr_shuffled(batch, 43)
        self.assertFalse(torch.equal(first['neighbor_lr_bias'], other['neighbor_lr_bias']))
        single = ab.lr_shuffled({key: value[1:2] for key, value in batch.items()}, 42)
        for key in ab.LR_BIAS_KEYS:
            self.assertTrue(torch.equal(single[key][0], first[key][1]))

    def test_lr_shuffle_restricted_to_model_edges_never_moves_masked_evidence(self):
        batch = toy_batch()
        present = batch['neighbor_present']
        edges = ab.present_pairs(present)
        self.assertEqual([int(row.sum()) for row in edges], [6, 6, 0])
        edges[0, 0, 2] = edges[0, 2, 0] = False  # neighbors 0 and 2 of center 0 lie beyond the model radius
        edges[1] = False
        edges[1, 0, 2] = True  # a single edge cannot be reassigned
        result = ab.lr_shuffled(batch, 42, present, edges)
        self.assertEqual(changed_keys(batch, result), set(ab.LR_BIAS_KEYS))
        old, new = batch['neighbor_lr_bias'], result['neighbor_lr_bias']
        self.assertTrue(torch.equal(old[~edges], new[~edges]))
        for index in range(len(present)):
            np.testing.assert_array_equal(np.sort(old[index][edges[index]].numpy()), np.sort(new[index][edges[index]].numpy()))
        self.assertTrue(bool((old[0][edges[0]] != new[0][edges[0]]).all()))
        self.assertTrue(torch.equal(old[1], new[1]))
        # What the model consumes, torch.where(eligible, bias, 0), keeps its total.
        self.assertAlmostEqual(float(torch.where(edges, old, 0.).double().sum()),
                               float(torch.where(edges, new, 0.).double().sum()), places=6)
        # The unrestricted default moves the masked entries; the center reassignment is the same.
        default = ab.lr_shuffled(batch, 42)
        self.assertTrue(torch.equal(default['center_lr_bias'], result['center_lr_bias']))
        self.assertNotEqual(float(default['neighbor_lr_bias'][0, 0, 2]), float(old[0, 0, 2]))
        self.assertEqual(float(new[0, 0, 2]), float(old[0, 0, 2]))
        through = ab.apply_ablation('lr_shuffled', batch, 42, (present, edges))
        self.assertEqual(changed_keys(result, through), set())
        self.assertEqual(changed_keys(ab.lr_off(batch), ab.apply_ablation('lr_off', batch, 42, (present, edges))), set())

    def test_lr_shuffle_rejects_edges_outside_present_pairs(self):
        batch = toy_batch()
        present = batch['neighbor_present']
        absent = ab.present_pairs(present)
        absent[0, 0, 3] = True  # slot 3 of center 0 is absent
        diagonal = ab.present_pairs(present)
        diagonal[0, 0, 0] = True
        for edges in (absent, diagonal, ab.present_pairs(present)[:, :3, :3]):
            with self.assertRaisesRegex(ValueError, 'present neighbors'):
                ab.lr_shuffled(batch, 42, present, edges)
        with self.assertRaisesRegex(ValueError, 'present neighbors'):
            ab.lr_shuffled(batch, 42, present[:, :3])

    def test_lr_reassignment_matches_hand_computation(self):
        present = torch.tensor([[True, True, True], [True, False, False]])
        bias = torch.tensor([[[0., 1., 2.], [3., 0., 4.], [5., 6., 0.]], [[0., 7., 0.], [0., 0., 0.], [0., 0., 0.]]])
        center = torch.tensor([[.25, .5, 0.], [.75, .875, .875]])
        batch = dict(neighbor_present=present, neighbor_lr_bias=bias, center_lr_bias=center,
                     cell_identity=torch.tensor([[0, 1], [0, 2]]))
        edges = ab.present_pairs(present)
        edges[0, 0, 2] = edges[0, 2, 0] = False
        shuffled = ab.lr_shuffled(batch, 42, present, edges)
        self.assertEqual((float(shuffled['neighbor_lr_bias'][0, 0, 2]), float(shuffled['neighbor_lr_bias'][0, 2, 0]),
                          float(shuffled['neighbor_lr_bias'][1, 0, 1])), (2., 5., 7.))
        info = ab.lr_reassignment(batch, shuffled, present, edges)
        self.assertEqual(info, dict(centers=2, present_neighbors=4, graph_edges=4, present_pairs_outside_graph=2,
            centers_without_movable_edges=1, edge_nonzero=4, edge_changed=4, edge_evidence_before=14.,
            edge_evidence_after=14., center_nonzero=3, center_changed=3, center_evidence_before=1.5,
            center_evidence_after=1.5))
        loose = ab.lr_reassignment(batch, ab.lr_shuffled(batch, 42))
        self.assertEqual((loose['graph_edges'], loose['present_pairs_outside_graph'], loose['edge_evidence_before']), (6, 0, 21.))
        total = ab.sum_counts([info, info])
        self.assertEqual((total['graph_edges'], total['edge_evidence_before'], total['centers']), (8, 28., 4))
        self.assertEqual(ab.sum_counts([]), {})

    def test_no_query_target_reaches_the_embedding_forward(self):
        batch = toy_batch()
        inputs = ab.embedding_inputs(batch)
        self.assertFalse(any(key.startswith('query_') for key in inputs))
        self.assertEqual(set(batch) - set(inputs), {key for key in batch if key.startswith('query_')})

    def test_targets_are_restored_from_the_real_batch_and_guarded(self):
        real = toy_batch()
        perturbed = clone(real)
        perturbed['query_nucleus_counts'] = real['query_total_counts'].clone()
        perturbed['query_cytoplasm_counts'] = torch.zeros_like(real['query_cytoplasm_counts'])
        restored = ab.restore_targets(perturbed, real)
        for key in ab.TARGET_KEYS:
            self.assertIs(restored[key], real[key])
        self.assertIs(restored['nucleus_bins'], perturbed['nucleus_bins'])
        wrong = clone(real)
        wrong['query_gene_ids'] = wrong['query_gene_ids'] + 1
        with self.assertRaisesRegex(ValueError, 'query_gene_ids'):
            ab.restore_targets(wrong, real)

    def test_swapped_targets_only_exchange_the_two_count_targets(self):
        batch = toy_batch()
        swapped = ab.swap_targets(batch)
        self.assertIs(swapped['query_nucleus_counts'], batch['query_cytoplasm_counts'])
        self.assertIs(swapped['query_cytoplasm_counts'], batch['query_nucleus_counts'])
        self.assertLessEqual(changed_keys(batch, swapped), {'query_nucleus_counts', 'query_cytoplasm_counts'})
        self.assertTrue(torch.equal(swapped['query_nucleus_counts'] + swapped['query_cytoplasm_counts'],
                                    batch['query_total_counts']))


class NeighborSubstitutionTests(unittest.TestCase):
    def setUp(self):
        self.eligible = np.ones(60, dtype=bool)
        self.eligible[[7, 20, 21]] = False
        self.centers = np.array([0, 5, 9, 30])
        self.neighbors = np.array([[1, 2, 7, 3], [6, 8, -1, -1], [10, 11, 12, 13], [-1, -1, -1, -1]])

    def test_presence_pattern_center_and_real_neighbors(self):
        result, info = ab.substitute_neighbors(self.neighbors, self.centers, self.eligible, seed=42, slice_seed=9)
        self.assertEqual(result.shape, self.neighbors.shape)
        np.testing.assert_array_equal(result < 0, self.neighbors < 0)
        self.assertEqual(result[0, 2], 7)  # an expression-free real neighbor stays, it is absent either way
        for center, real, row in zip(self.centers, self.neighbors, result):
            drawn = row[(real >= 0) & self.eligible[np.maximum(real, 0)]]
            self.assertNotIn(center, row)
            self.assertEqual(len(np.unique(row[row >= 0])), int((row >= 0).sum()))
            self.assertTrue(self.eligible[drawn].all())
            self.assertFalse(np.isin(drawn, real).any())
        np.testing.assert_array_equal(self.eligible[np.maximum(result, 0)] & (result >= 0),
                                      self.eligible[np.maximum(self.neighbors, 0)] & (self.neighbors >= 0))
        self.assertEqual(info, dict(centers=4, substituted_slots=9, retained_absent_slots=1, relaxed_centers=0,
                                    unchanged_slots=0))

    def test_deterministic_seeded_and_independent_of_batch_layout(self):
        first, _ = ab.substitute_neighbors(self.neighbors, self.centers, self.eligible, seed=42, slice_seed=9)
        second, _ = ab.substitute_neighbors(self.neighbors.copy(), self.centers, self.eligible, seed=42, slice_seed=9)
        np.testing.assert_array_equal(first, second)
        part, _ = ab.substitute_neighbors(self.neighbors[2:3], self.centers[2:3], self.eligible, seed=42, slice_seed=9)
        np.testing.assert_array_equal(part[0], first[2])
        other, _ = ab.substitute_neighbors(self.neighbors, self.centers, self.eligible, seed=43, slice_seed=9)
        self.assertFalse(np.array_equal(first, other))
        slide, _ = ab.substitute_neighbors(self.neighbors, self.centers, self.eligible, seed=42, slice_seed=10)
        self.assertFalse(np.array_equal(first, slide))

    def test_tiny_slide_falls_back_to_real_neighbors_but_never_the_center(self):
        result, info = ab.substitute_neighbors(np.array([[1, 2, 3]]), np.array([0]), np.ones(4, dtype=bool))
        np.testing.assert_array_equal(np.sort(result[0]), [1, 2, 3])
        self.assertEqual(info['relaxed_centers'], 1)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Center leaked'):
            ab.substitute_neighbors(np.array([[0, 1]]), np.array([0]), np.ones(4, dtype=bool))
        with self.assertRaises(ValueError):
            ab.substitute_neighbors(np.array([[1, 9]]), np.array([0]), np.ones(4, dtype=bool))

    def test_expressed_cells_follow_the_encoder_presence_rule(self):
        nuclear = np.array([[0, 0], [1, 0], [0, 0]], dtype=np.uint32)
        cyto = np.array([[0, 0], [0, 0], [0, 2]], dtype=np.uint32)
        np.testing.assert_array_equal(ab.expressed_cells(nuclear, cyto), [False, True, True])


class EncoderPathTests(unittest.TestCase):
    """Data-level ablations through the unchanged V12 encoder on a 14-cell synthetic slide."""

    def setUp(self):
        from spatial_pretrain.data.subcellular_v12 import prepare_view_batch_v12
        self.reference = prepare_view_batch_v12
        rng = np.random.default_rng(11)
        self.genes = np.array([5, 7, 9, 12, 21, 30], dtype=np.int64)
        self.nuclear = rng.integers(0, 6, (14, 6)).astype(np.uint32)
        self.cyto = rng.integers(0, 6, (14, 6)).astype(np.uint32)
        self.nuclear[:, 0] += 1
        self.cyto[:, 1] += 1
        self.nuclear[13] = 0
        self.cyto[13] = 0
        self.coords = (rng.random((14, 2)) * 40).astype(np.float32)
        self.coords[13] = self.coords[0] + np.float32(.01)  # nearest real neighbor of center 0 has no expression
        self.centers = np.array([0, 4, 8])
        self.neighbors = ev12.real_neighbors(self.coords, self.centers, 3)
        self.options = dict(pad_token_id=0, cls_token_id=1, max_tokens=8, query_genes=2, seed=42, epoch=0,
            slice_seed=3, slice_index=1, lr_pairs=np.array([[5, 7], [9, 12], [21, 5]]),
            lr_weights=np.array([1., .5, .25], dtype=np.float32), max_neighbor_distance=100.)
        self.passes = (dict(mask=False), dict(mask=True, gene_mask_ratio=0., neighbor_dropout=0.))

    def real(self, extra, nuclear=None, cyto=None, centers=None, neighbors=None, coords=None):
        return self.reference(self.nuclear if nuclear is None else nuclear, self.cyto if cyto is None else cyto,
            self.genes, self.centers if centers is None else centers, self.neighbors if neighbors is None else neighbors,
            self.coords if coords is None else coords, **self.options, **extra)

    def test_fixture_contains_an_absent_real_neighbor(self):
        self.assertIn(13, self.neighbors[0])
        self.assertFalse(bool(self.real(self.passes[0])['neighbor_present'][0].all()))

    def test_equal_content_and_coordinates_reproduce_the_reference_batch(self):
        for extra in self.passes:
            mine = ab.prepare_batch(self.nuclear, self.cyto, self.genes, self.centers, self.neighbors, self.neighbors,
                                    self.coords, **self.options, **extra)
            self.assertEqual(changed_keys(self.real(extra), mine), set())

    def test_substitution_equals_moving_random_cells_into_the_real_positions(self):
        eligible = ab.expressed_cells(self.nuclear, self.cyto)
        substitutes, info = ab.substitute_neighbors(self.neighbors, self.centers, eligible, seed=42, slice_seed=3)
        self.assertGreaterEqual(info['retained_absent_slots'], 1)
        self.assertEqual(info['retained_absent_slots'], int((self.neighbors == 13).sum()))
        self.assertEqual(info['relaxed_centers'], 0)
        np.testing.assert_array_equal(substitutes == 13, self.neighbors == 13)
        for extra in self.passes:
            real = self.real(extra)
            shuffled = ab.prepare_batch(self.nuclear, self.cyto, self.genes, self.centers, substitutes, self.neighbors,
                                        self.coords, **self.options, **extra)
            ab.check_substitution(real, shuffled)
            self.assertTrue(torch.equal(real['neighbor_present'], shuffled['neighbor_present']))
            self.assertTrue(torch.equal(real['relative_coords'], shuffled['relative_coords']))
            for index, center in enumerate(self.centers):
                present = shuffled['neighbor_present'][index].numpy()
                nodes = shuffled['neighbor_cell_identity'][index, :, 1].numpy()[present]
                self.assertNotIn(center, nodes)
                self.assertFalse(np.isin(nodes, self.neighbors[index]).any())
                self.assertEqual(set(nodes), set(substitutes[index][eligible[substitutes[index]]]))
                # Same result as the reference path on a slide where the substitutes physically sit
                # at the real neighbors' coordinates: tokens AND both LR matrices are recomputed.
                moved = self.coords.copy()
                moved[substitutes[index]] = self.coords[self.neighbors[index]]
                expected = self.real(extra, centers=self.centers[index:index + 1],
                                     neighbors=substitutes[index:index + 1], coords=moved)
                single = ab.prepare_batch(self.nuclear, self.cyto, self.genes, self.centers[index:index + 1],
                    substitutes[index:index + 1], self.neighbors[index:index + 1], self.coords, **self.options, **extra)
                self.assertEqual(changed_keys(expected, single), set())
            self.assertFalse(torch.equal(real['neighbor_lr_bias'], shuffled['neighbor_lr_bias']))

    def test_substitution_guard_detects_changed_presence_or_center(self):
        real = self.real(self.passes[0])
        broken = dict(real)
        broken['neighbor_present'] = ~real['neighbor_present']
        with self.assertRaisesRegex(ValueError, 'neighbor_present'):
            ab.check_substitution(real, broken)
        broken = dict(real)
        identities = real['neighbor_cell_identity'].clone()
        row, slot = torch.nonzero(real['neighbor_present'])[0].tolist()
        identities[row, slot] = real['cell_identity'][row]
        broken['neighbor_cell_identity'] = identities
        with self.assertRaisesRegex(ValueError, 'own substitute'):
            ab.check_substitution(real, broken)

    def test_content_and_coordinate_rows_must_share_padding(self):
        wrong = self.neighbors.copy()
        wrong[0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'padding'):
            ab.prepare_batch(self.nuclear, self.cyto, self.genes, self.centers, wrong, self.neighbors, self.coords,
                             **self.options, mask=False)

    def test_whole_as_nucleus_uses_the_encoder_binning_and_keeps_true_targets(self):
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        whole = self.nuclear.astype(np.int64) + self.cyto
        for extra in self.passes:
            real = self.real(extra)
            mine = ab.prepare_batch(self.nuclear, self.cyto, self.genes, self.centers, self.neighbors, self.neighbors,
                self.coords, center_counts=ab.whole_as_nucleus_counts, **self.options, **extra)
            expected = self.real(extra, nuclear=whole, cyto=np.zeros_like(whole))
            self.assertEqual(changed_keys(expected, mine), set())
            self.assertEqual(int(mine['cytoplasm_present'].sum()), 0)
            self.assertEqual(float(mine['center_statistics'][:, [1, 3]].abs().sum()), 0.)
            unchanged = [key for key in real if key.startswith('neighbor_') or key.startswith('center_lr')]
            unchanged += ['relative_coords', 'cell_identity', 'query_gene_ids', 'query_valid_mask', 'query_total_counts']
            for key in unchanged:
                self.assertTrue(torch.equal(real[key], mine[key]), key)
            self.assertTrue(torch.equal(mine['query_nucleus_counts'], real['query_total_counts']))
            restored = ab.restore_targets(mine, real)
            for key in ab.TARGET_KEYS:
                self.assertTrue(torch.equal(restored[key], real[key]))
        np.testing.assert_array_equal(nuclear, self.nuclear)
        np.testing.assert_array_equal(cyto, self.cyto)

    def test_whole_center_counts_do_not_overflow_small_unsigned_types(self):
        nuclear = np.array([[200, 1], [3, 4]], dtype=np.uint8)
        cyto = np.array([[100, 2], [5, 6]], dtype=np.uint8)
        first, second = ab.whole_as_nucleus_counts(nuclear, cyto)
        np.testing.assert_array_equal(first, [[300, 3], [3, 4]])
        np.testing.assert_array_equal(second, [[0, 0], [5, 6]])
        np.testing.assert_array_equal(first[1] + second[1], nuclear[1].astype(int) + cyto[1])

    def test_batch_level_swap_equals_swapping_the_raw_compartments(self):
        """Also proves that center LR needs no recomputation under the swap."""
        targets = {'query_nucleus_counts', 'query_cytoplasm_counts'}
        for extra in self.passes:
            real = self.real(extra)
            swapped = ab.swap_compartments(real)
            dataset = self.real(extra, nuclear=self.cyto, cyto=self.nuclear)
            self.assertLessEqual(changed_keys(dataset, swapped), targets)
            self.assertTrue(torch.equal(dataset['query_nucleus_counts'], real['query_cytoplasm_counts']))
            for key in targets:  # the ablation leaves targets as they were
                self.assertIs(swapped[key], real[key])
            self.assertTrue(torch.equal(real['center_lr_bias'], dataset['center_lr_bias']))


class ModelPresenceAndLRTests(unittest.TestCase):
    """Assumptions about the unchanged spatial encoder that the ablations rely on."""

    def setUp(self):
        from spatial_pretrain.models.subcellular_v12 import LRNeighborEncoderV12
        torch.manual_seed(5)
        self.encoder = LRNeighborEncoderV12(16, 2, coordinate_scale=50., spatial_radius=100., graph_neighbors=8,
            lr_enabled=True, lr_max_strength=2., lr_initial_strength=1., dropout=0.).eval()
        self.batch = dict(neighbor_present=torch.tensor([[True, True, True, False], [True, True, True, True]]),
            relative_coords=torch.randn(2, 4, 2) * 20, neighbor_lr_bias=torch.rand(2, 4, 4), center_lr_bias=torch.rand(2, 4),
            cell_identity=torch.tensor([[0, 1], [0, 2]]))
        self.features = torch.randn(2, 4, 16)

    def run_encoder(self, batch):
        with torch.no_grad():
            pure, conditioned = self.encoder(self.features, batch['neighbor_present'], batch['relative_coords'],
                                             batch['neighbor_lr_bias'], batch['center_lr_bias'])[:2]
        return pure, conditioned

    def test_zeroed_bias_equals_zero_lr_strength(self):
        pure, conditioned = self.run_encoder(self.batch)
        off_pure, off_conditioned = self.run_encoder(ab.lr_off(self.batch))
        torch.testing.assert_close(off_pure, off_conditioned)
        self.assertFalse(torch.allclose(pure, off_pure, atol=1e-6))
        self.assertFalse(torch.allclose(conditioned, off_conditioned, atol=1e-6))
        with torch.no_grad():
            self.encoder.lr_scale_logit.fill_(-float('inf'))
        self.assertEqual(float(self.encoder.lr_strength.abs().sum()), 0.)
        zero_pure, zero_conditioned = self.run_encoder(self.batch)
        torch.testing.assert_close(off_pure, zero_pure)
        torch.testing.assert_close(off_conditioned, zero_conditioned)

    def test_shuffled_bias_changes_the_spatial_readouts(self):
        pure, conditioned = self.run_encoder(self.batch)
        moved_pure, moved_conditioned = self.run_encoder(ab.lr_shuffled(self.batch, 42))
        self.assertFalse(torch.allclose(pure, moved_pure, atol=1e-6))
        self.assertFalse(torch.allclose(conditioned, moved_conditioned, atol=1e-6))

    def test_shuffle_on_the_model_graph_preserves_the_evidence_the_encoder_reads(self):
        from types import SimpleNamespace
        coords = torch.zeros(2, 4, 2)
        coords[0, :, 0] = torch.tensor([-80., 0., 80., 0.])  # slots 0 and 2 are 160 um apart, slot 3 is absent
        coords[1, :, 0] = torch.tensor([0., 10., 20., 30.])
        batch = dict(self.batch, relative_coords=coords)
        model = SimpleNamespace(spatial_encoder=self.encoder,
                                _v12_presence=lambda item: (None, None, item['neighbor_present'].bool()))
        present, edges = ab.model_graph(model, batch)
        self.assertTrue(torch.equal(present, batch['neighbor_present']))
        self.assertEqual((int(edges[0].sum()), int(edges[1].sum())), (4, 12))
        self.assertFalse(bool(edges[0, 0, 2] | edges[0, 2, 0]))
        old = batch['neighbor_lr_bias']
        # The encoder ignores evidence outside its graph ...
        masked = dict(batch, neighbor_lr_bias=torch.where(edges, old, .5))
        for first, second in zip(self.run_encoder(batch), self.run_encoder(masked)):
            torch.testing.assert_close(first, second)
        # ... so the reassignment stays inside it and keeps the consumed total.
        shuffled = ab.lr_shuffled(batch, 42, present, edges)
        new = shuffled['neighbor_lr_bias']
        self.assertTrue(torch.equal(old[~edges], new[~edges]))
        self.assertAlmostEqual(float(torch.where(edges, old, 0.).double().sum()),
                               float(torch.where(edges, new, 0.).double().sum()), places=6)
        info = ab.lr_reassignment(batch, shuffled, present, edges)
        self.assertEqual((info['graph_edges'], info['present_pairs_outside_graph'], info['edge_changed']), (16, 2, 16))
        self.assertAlmostEqual(info['edge_evidence_before'], info['edge_evidence_after'], places=6)
        pure, _ = self.run_encoder(batch)
        moved, _ = self.run_encoder(shuffled)
        self.assertFalse(torch.allclose(pure, moved, atol=1e-6))
        # Cycling over every present pair instead leaks evidence across the radius mask.
        loose = ab.lr_shuffled(batch, 42)['neighbor_lr_bias']
        self.assertNotEqual(float(loose[0, 0, 2]), float(old[0, 0, 2]))

    def test_dropped_neighbors_give_exactly_zero_spatial_readouts(self):
        pure, conditioned = self.run_encoder(ab.drop_neighbors(self.batch))
        self.assertEqual(float(pure.abs().sum()), 0.)
        self.assertEqual(float(conditioned.abs().sum()), 0.)


def probe_row(ablation, embedding, task, value):
    row = dict(ablation=ablation, embedding=embedding, task=task, embedding_dimension=512, train_cells=4000, val_cells=1200)
    if task in ev12.FIXED_C:
        row['fixed_C_result'] = dict(C=ev12.FIXED_C[task], val_macro_f1=value, val_balanced_accuracy=value / 2,
                                     per_donor_macro_f1=dict(A=value + .1, B=value - .1))
    else:
        row.update(ridge_alpha=1., mean_val_label_coverage=.9, val_result=dict(mae=value / 10, js_divergence=value ** 2, js_distance=value))
    return row


def count_row(ablation, mode, split, nll, donor='pooled', orientation='as_is'):
    return dict(ablation=ablation, target_orientation=orientation, split=split, mode=mode, donor=donor, query_count=10,
                positive_query_count=4, available_cells=2, nll=nll, total_abs_error=.3, nucleus_abs_error=.2,
                cytoplasm_abs_error=.1, allocation_abs_error=None if nll is None else .4)


COMMON = dict(variant='v', step=3400, checkpoint_sha256='c', config_sha256='g', cohort_sha256='h', metrics_path='m')


class MetricTableTests(unittest.TestCase):
    def test_ablation_list_is_canonical_and_always_starts_with_none(self):
        self.assertEqual(ab.parse_ablations(['all']), list(ab.ABLATIONS))
        self.assertEqual(ab.parse_ablations(['lr_off,drop_neighbors', 'lr_off']), ['none', 'drop_neighbors', 'lr_off'])
        self.assertEqual(ab.parse_ablations(['none']), ['none'])
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            ab.parse_ablations(['lr_of'])
        self.assertEqual(set(ab.DESCRIPTIONS), set(ab.ABLATIONS))
        self.assertEqual(ab.READOUTS, ('nucleus', 'cytoplasm', 'local', 'spatial', 'fused', 'concat', 'concat_pca512'))

    def test_embedding_shift_matches_hand_computation(self):
        reference = np.array([[1., 0.], [0., 2.], [0., 0.]])
        perturbed = np.array([[0., 1.], [0., 2.], [1., 0.]])
        result = ab.embedding_shift(reference, perturbed)
        self.assertEqual((result['cells'], result['compared_cells']), (3, 2))
        self.assertAlmostEqual(result['mean_cosine'], .5)
        self.assertAlmostEqual(result['min_cosine'], 0.)
        self.assertAlmostEqual(result['mean_relative_l2'], np.sqrt(2) / 2)
        self.assertAlmostEqual(result['max_abs_difference'], 1.)
        self.assertEqual((result['zero_reference_cells'], result['zero_perturbed_cells']), (1, 0))
        same = ab.embedding_shift(reference, reference)
        self.assertAlmostEqual(same['mean_cosine'], 1.)
        self.assertEqual(same['max_abs_difference'], 0.)
        empty = ab.embedding_shift(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertIsNone(empty['mean_cosine'])
        self.assertIsNone(empty['mean_relative_l2'])
        with self.assertRaises(ValueError):
            ab.embedding_shift(reference, perturbed[:2])

    def test_weighted_diagnostics(self):
        result = ab.weighted_diagnostics([(2, dict(a=1.)), (6, dict(a=3., b=5.))])
        self.assertAlmostEqual(result['a'], 2.5)
        self.assertAlmostEqual(result['b'], 5.)
        self.assertEqual(ab.weighted_diagnostics([]), {})

    def test_flat_rows_use_the_reference_comparison_columns(self):
        row = ab.flatten_probe_row(COMMON, probe_row('lr_off', 'fused', 'region', .3))
        self.assertEqual(list(row)[:6], ['variant', 'step', 'checkpoint_sha256', 'config_sha256', 'cohort_sha256', 'metrics_path'])
        self.assertEqual((row['ablation'], row['embedding'], row['task'], row['dimension']), ('lr_off', 'fused', 'region', 512))
        self.assertEqual(row['probe_fit'], ab.REFIT)
        self.assertEqual((row['C'], row['val_macro_f1'], row['val_balanced_accuracy']), (.01, .3, .15))
        self.assertAlmostEqual(row['donor_macro_f1_A'], .4)
        row = ab.flatten_probe_row(COMMON, probe_row('none', 'spatial', 'neighbor_composition', .5))
        self.assertEqual((row['val_mae'], row['val_js_divergence'], row['val_js_distance'], row['ridge_alpha']), (.05, .25, .5, 1.))
        self.assertNotIn('val_macro_f1', row)
        skipped = ab.flatten_probe_row(COMMON, dict(ablation='drop_neighbors', embedding='spatial', task='region',
                                                    embedding_dimension=512, skipped='constant readout'))
        self.assertEqual(skipped['skipped'], 'constant readout')
        count = ab.flatten_count_row(COMMON, count_row('none', 'fused', 'val', .6))
        self.assertEqual((count['embedding'], count['task'], count['mode'], count['nll']), ('fused', ab.COUNT_TASK, 'fused', .6))

    def flat(self):
        probes = [probe_row('none', 'fused', 'region', .30), probe_row('lr_off', 'fused', 'region', .25),
                  probe_row('none', 'spatial', 'neighbor_composition', .40),
                  probe_row('drop_neighbors', 'spatial', 'neighbor_composition', .55),
                  probe_row(ab.BASELINE_LABEL, 'scgpt_whole', 'region', .28)]
        counts = [count_row('none', 'fused', 'val', .60), count_row('lr_off', 'fused', 'val', .65),
                  count_row('lr_off', 'fused', 'val', 9., donor='A'), count_row('drop_neighbors', 'spatial', 'val', None),
                  count_row('none', 'spatial', 'val', .9), count_row('swap_compartments', 'fused', 'val', 7., orientation='swapped')]
        return ([ab.flatten_probe_row(COMMON, row) for row in probes]
                + [ab.flatten_count_row(COMMON, row) for row in counts])

    def test_delta_table_matches_hand_computation(self):
        deltas = {(row['ablation'], row['embedding'], row['task'], row['split'], row['metric']): row
                  for row in ab.delta_vs_none(self.flat())}
        row = deltas[('lr_off', 'fused', 'region', None, 'val_macro_f1')]
        self.assertAlmostEqual(row['delta'], -.05)
        self.assertTrue(row['higher_is_better'])
        row = deltas[('drop_neighbors', 'spatial', 'neighbor_composition', None, 'val_js_distance')]
        self.assertAlmostEqual(row['delta'], .15)
        self.assertFalse(row['higher_is_better'])
        row = deltas[('lr_off', 'fused', ab.COUNT_TASK, 'val', 'nll')]  # pooled .65, never donor A's 9.
        self.assertAlmostEqual(row['delta'], .05)
        self.assertEqual((row['none'], row['value']), (.60, .65))
        row = deltas[('drop_neighbors', 'spatial', ab.COUNT_TASK, 'val', 'nll')]  # unavailable view: no fabricated delta
        self.assertIsNone(row['delta'])
        self.assertEqual(row['none'], .9)
        self.assertFalse(any(key[0] in ('none', ab.BASELINE_LABEL, 'swap_compartments') for key in deltas))

    def test_transfer_rows_get_their_own_delta_against_the_refit_none_reference(self):
        transfer = dict(probe_row('lr_off', 'fused', 'region', .10), probe_fit=ab.TRANSFER)
        flat = self.flat() + [ab.flatten_probe_row(COMMON, transfer)]
        self.assertEqual((flat[0]['probe_fit'], flat[-1]['probe_fit']), (ab.REFIT, ab.TRANSFER))
        key = ('lr_off', 'fused', 'region', None, 'val_macro_f1')
        self.assertEqual(ab.measured_values(flat)[key], .25)  # never overwritten by the transfer row
        self.assertEqual(ab.measured_values(flat), ab.measured_values(self.flat()))
        moved = ab.measured_values(flat, ab.TRANSFER)
        self.assertEqual(set(moved), {key, key[:4] + ('val_balanced_accuracy',)})
        self.assertEqual(moved[key], .10)
        deltas = {(row['ablation'], row['probe_fit'], row['embedding'], row['task'], row['metric']): row
                  for row in ab.delta_vs_none(flat)}
        self.assertAlmostEqual(deltas[('lr_off', ab.REFIT, 'fused', 'region', 'val_macro_f1')]['delta'], -.05)
        row = deltas[('lr_off', ab.TRANSFER, 'fused', 'region', 'val_macro_f1')]
        self.assertAlmostEqual(row['delta'], -.20)
        self.assertEqual((row['none'], row['value']), (.30, .10))
        self.assertAlmostEqual(deltas[('lr_off', None, 'fused', ab.COUNT_TASK, 'nll')]['delta'], .05)
        self.assertEqual(len(ab.delta_vs_none(flat)), len(ab.delta_vs_none(self.flat())) + 2)

    def test_hybrid_features_keep_train_rows_unperturbed(self):
        reference = np.arange(12, dtype=np.float32).reshape(4, 3)
        perturbed = -reference - 1
        train = np.array([True, False, True, False])
        hybrid = ab.hybrid_features(reference, perturbed, train)
        self.assertEqual((hybrid.dtype, hybrid.shape), (np.float32, (4, 3)))
        np.testing.assert_array_equal(hybrid[train], reference[train])
        np.testing.assert_array_equal(hybrid[~train], perturbed[~train])
        # Everything the reference probes fit comes from train rows: no perturbed value can reach it.
        other = ab.hybrid_features(reference, perturbed * 7, train)
        np.testing.assert_array_equal(hybrid[train], other[train])
        np.testing.assert_array_equal(ab.hybrid_features(reference, reference, train), reference)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            ab.hybrid_features(reference, perturbed[:3], train)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            ab.hybrid_features(reference, perturbed, train.astype(int))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            ab.hybrid_features(reference, perturbed, train[:3])

    def test_unperturbed_path_anchor_coverage(self):
        wanted = len(ab.READOUTS) * len(ev12.FIXED_C) + len(ev12.COUNT_MODES)
        missing = ab.none_anchor_coverage(ab.builtin_anchors(ab.OFFICIAL_VARIANT, 100))  # baseline anchors only
        self.assertEqual((missing['status'], missing['enforced_checks'], len(missing['unanchored'])), ('unanchored', 0, wanted))
        partial = ab.none_anchor_coverage(ab.builtin_anchors(ab.OFFICIAL_VARIANT, 3400))
        self.assertEqual((partial['status'], partial['enforced_checks'], len(partial['unanchored'])), ('partial', 11, wanted - 9))
        self.assertIn('cytoplasm/cell_type', partial['unanchored'])
        self.assertIn(f'nucleus/{ab.COUNT_TASK}', partial['unanchored'])
        self.assertNotIn('fused/region', partial['unanchored'])
        self.assertNotIn(f'fused/{ab.COUNT_TASK}', partial['unanchored'])
        full = [ab.anchor('reference_metrics', 'none', name, task, 'val_macro_f1', .5)
                for name in ab.READOUTS for task in ev12.FIXED_C]
        full += [ab.anchor('reference_metrics', 'none', mode, ab.COUNT_TASK, 'nll', .6, split='val') for mode in ev12.COUNT_MODES]
        self.assertEqual(ab.none_anchor_coverage(full), dict(status='full', enforced_checks=wanted, unanchored=[]))
        informational = [dict(item, enforced=False) for item in full]
        self.assertEqual(ab.none_anchor_coverage(informational)['status'], 'unanchored')

    def test_only_launcher_logs_may_precede_the_run_contract(self):
        self.assertEqual(ab.unexpected_entries([]), [])
        self.assertEqual(ab.unexpected_entries(['logs', 'slurm-123.out', 'run.log', 'stderr.err']), [])
        self.assertEqual(ab.unexpected_entries(['run.log', 'none', 'metrics.csv', 'progress.json']),
                         ['metrics.csv', 'none', 'progress.json'])

    def test_builtin_anchors_cover_the_official_numbers(self):
        official = ab.builtin_anchors(ab.OFFICIAL_VARIANT, 3400)
        self.assertEqual(len(official), 6 + 8 + 3)
        self.assertTrue(all(item['enforced'] for item in official))
        lookup = {(item['ablation'], item['embedding'], item['task'], item['split'], item['metric']): item['expected']
                  for item in official}
        self.assertEqual(lookup[(ab.BASELINE_LABEL, 'scgpt_whole', 'neighbor_composition', None, 'val_js_distance')], .381156)
        self.assertEqual(lookup[('none', 'fused', 'region', None, 'val_macro_f1')], .373513)
        self.assertEqual(lookup[('none', 'fused', ab.COUNT_TASK, 'val', 'nll')], .6006)
        self.assertEqual(len(ab.builtin_anchors(ab.OFFICIAL_VARIANT, 0)), 6 + 6)
        self.assertEqual(len(ab.builtin_anchors(ab.OFFICIAL_VARIANT, 100)), 6)
        self.assertEqual(len(ab.builtin_anchors('another_run', 3400)), 6)

    def test_anchor_comparison_fails_only_beyond_tolerance(self):
        anchors = [ab.anchor('t', 'none', 'fused', 'region', 'val_macro_f1', .2981),
                   ab.anchor('t', 'none', 'fused', ab.COUNT_TASK, 'nll', .6019, split='val')]
        measured = ab.measured_values(self.flat())
        self.assertEqual(measured[('none', 'fused', 'region', None, 'val_macro_f1')], .30)
        self.assertEqual(measured[('lr_off', 'fused', ab.COUNT_TASK, 'val', 'nll')], .65)
        result = ab.compare_anchors(anchors, measured)
        self.assertTrue(result['passed'])
        self.assertEqual((result['failures'], result['enforced_checks'], result['tolerance']), (0, 2, 2e-3))
        self.assertAlmostEqual(result['checks'][0]['abs_diff'], .0019)
        failed = ab.compare_anchors([ab.anchor('t', 'none', 'fused', 'region', 'val_macro_f1', .2979)], measured)
        self.assertFalse(failed['passed'])
        self.assertEqual(failed['failures'], 1)
        missing = ab.compare_anchors([ab.anchor('t', 'none', 'local', 'region', 'val_macro_f1', .3)], measured)
        self.assertFalse(missing['passed'])
        self.assertIsNone(missing['checks'][0]['measured'])
        unavailable = ab.compare_anchors([ab.anchor('t', 'drop_neighbors', 'spatial', ab.COUNT_TASK, 'nll', .5, split='val')], measured)
        self.assertFalse(unavailable['passed'])
        informational = ab.compare_anchors([ab.anchor('t', 'none', 'fused', 'region', 'val_macro_f1', .9, enforced=False)], measured)
        self.assertTrue(informational['passed'])
        self.assertFalse(informational['checks'][0]['within_tolerance'])

    def test_reference_metrics_anchor_every_probe_f1_and_pooled_nll(self):
        results = [{key: value for key, value in probe_row('x', name, task, .4).items() if key != 'ablation'}
                   for name in ('fused', 'scgpt_whole') for task in ev12.TASKS]
        counts = [dict(split='val', mode='fused', donor='pooled', nll=.6, total_abs_error=.4, nucleus_abs_error=.2,
                       cytoplasm_abs_error=.2, allocation_abs_error=None),
                  dict(split='val', mode='fused', donor='A', nll=.7, total_abs_error=.4, nucleus_abs_error=.2,
                       cytoplasm_abs_error=.2, allocation_abs_error=.3)]
        report = dict(status='complete', model_version=12, step=3400, checkpoint_sha256='abc',
                      cohort_sha256=ev12.CANONICAL_COHORT, results=results, masked_count_prediction=dict(results=counts))
        anchors = ab.reference_anchors(report, 3400, 'abc')
        enforced = {(item['ablation'], item['embedding'], item['task'], item['split'], item['metric'])
                    for item in anchors if item['enforced']}
        self.assertEqual(enforced, {('none', 'fused', 'cell_type', None, 'val_macro_f1'),
            ('none', 'fused', 'region', None, 'val_macro_f1'),
            (ab.BASELINE_LABEL, 'scgpt_whole', 'cell_type', None, 'val_macro_f1'),
            (ab.BASELINE_LABEL, 'scgpt_whole', 'region', None, 'val_macro_f1'),
            ('none', 'fused', ab.COUNT_TASK, 'val', 'nll')})
        informational = {(item['embedding'], item['metric']) for item in anchors if not item['enforced']}
        self.assertIn(('fused', 'val_js_distance'), informational)
        self.assertIn(('fused', 'total_abs_error'), informational)
        self.assertNotIn(('fused', 'allocation_abs_error'), informational)
        for step, digest in ((3300, 'abc'), (3400, 'other')):
            with self.assertRaisesRegex(ValueError, 'different checkpoint'):
                ab.reference_anchors(report, step, digest)
        with self.assertRaisesRegex(ValueError, 'cohort'):
            ab.reference_anchors(dict(report, cohort_sha256='x'), 3400, 'abc')
        with self.assertRaisesRegex(ValueError, 'complete'):
            ab.reference_anchors(dict(report, status='running'), 3400, 'abc')

    def test_csv_is_atomic_and_keeps_the_union_of_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            ab.write_csv(path, self.flat())
            self.assertFalse(path.with_name('metrics.csv.partial').exists())
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0]['ablation'], 'none')
        self.assertEqual(rows[0]['nll'], '')
        self.assertEqual(rows[-1]['target_orientation'], 'swapped')
        self.assertEqual(rows[8]['nll'], '')  # unavailable view stays empty, never zero

    def test_count_frame_attaches_identities_and_rejects_duplicates(self):
        import pandas as pd
        obs = pd.DataFrame(dict(full_cell_id=['a', 'b', 'c'], patient=['p', 'p', 'p']))
        rows = [dict(center_row=2, mode='fused', nll_sum=1.), dict(center_row=0, mode='fused', nll_sum=2.)]
        frame = ab.count_frame(rows, obs, 'val')
        self.assertEqual(frame.full_cell_id.tolist(), ['c', 'a'])
        self.assertEqual(frame['split'].tolist(), ['val', 'val'])
        with self.assertRaisesRegex(ValueError, 'identities'):
            ab.count_frame(rows + rows[:1], obs, 'val')


if __name__ == '__main__':
    unittest.main()
