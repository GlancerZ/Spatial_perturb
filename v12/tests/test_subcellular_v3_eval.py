"""CPU-only V3 evaluation QA: relative targets, masks, dimensions and provenance."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location('v3_evaluator_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/evaluate_subcellular_v3.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def diagnostic_batch():
    batch = dict(
        token_valid_mask=torch.tensor([[[True, True, True]], [[True, True, False]]]),
        compartment_ids=torch.tensor([[[2, 0, 1]], [[2, 0, 2]]]),
        query_compartment_ids=torch.tensor([[0, 1, 1], [0, 1, 1]]),
        query_valid_mask=torch.tensor([[True, True, False], [True, True, False]]),
        target_bins=torch.tensor([[0, 50, -2], [25, 0, -2]]))
    prediction = torch.tensor([[[-2., 0.], [2., 0.], [99., 99.]],
                               [[2., 0.], [-2., 0.], [99., 99.]]])
    return batch, prediction


class RelativeDiagnosticTests(unittest.TestCase):
    def test_padding_is_excluded_and_positive_rank_is_normalized(self):
        batch, prediction = diagnostic_batch()
        metrics = module.decoder_metrics(prediction, batch, batch['query_valid_mask'])
        self.assertEqual(metrics['queries'], 4)
        self.assertEqual(metrics['positive_queries'], 2)
        self.assertEqual(metrics['nonzero_balanced_accuracy'], 1.)
        self.assertEqual(metrics['nonzero_auroc'], 1.)
        expected = (.25 + (.5 - 24. / 49.) ** 2) / 2.
        self.assertAlmostEqual(metrics['positive_rank_mse'], expected, places=7)
        self.assertAlmostEqual(metrics['positive_bin_mae'], 12.5, places=6)
        changed = dict(batch)
        changed['target_bins'] = batch['target_bins'].clone()
        changed['target_bins'][~batch['query_valid_mask']] = 10000
        self.assertEqual(module.decoder_metrics(prediction, changed, changed['query_valid_mask']), metrics)

    def test_compartment_context_mask_matches_training_loss_and_skips_empty_cytoplasm(self):
        batch, prediction = diagnostic_batch()
        mask = module.compartment_query_mask(batch)
        self.assertEqual(mask.tolist(), [[True, True, True], [True, False, False]])
        output = dict(compartment_pred=prediction, joint_pred=prediction)
        report = module.aggregate_masked_predictions(output, batch, {'loss': {'comp': 1., 'joint': 1.}}, 100)
        self.assertEqual(report['decoders']['compartment']['queries'], 3)
        self.assertEqual(report['decoders']['joint']['queries'], 4)
        self.assertAlmostEqual(report['loss_mean']['total'],
                               report['loss_mean']['compartment'] + report['loss_mean']['joint'], places=6)
        self.assertTrue(torch.isfinite(torch.tensor(list(report['loss_mean'].values()))).all())

    def test_invalid_bin_is_rejected_only_when_query_is_valid(self):
        batch, prediction = diagnostic_batch()
        batch['target_bins'][0, 0] = -2
        with self.assertRaisesRegex(ValueError, 'integer bins'):
            module.decoder_metrics(prediction, batch, batch['query_valid_mask'])

    def test_empty_query_subset_has_null_metrics_instead_of_nan(self):
        batch, prediction = diagnostic_batch()
        result = module.decoder_metrics(prediction, batch, torch.zeros_like(batch['query_valid_mask']))
        self.assertEqual(result['queries'], 0)
        self.assertIsNone(result['nonzero_bce'])
        self.assertIsNone(result['positive_rank_mse'])
        self.assertIsNone(result['nonzero_auroc'])

    def test_dual_cls_dimensions_do_not_reuse_v2_512_dimension_assumption(self):
        result = module.expected_dimensions(True)
        self.assertEqual(result, {'base': 1024, 'intrinsic': 1024, 'joint': 1024,
                                  'nucleus': 512, 'cytoplasm': 512, 'joint_neighbor_shuffle': 1024})


class SourceProvenanceTests(unittest.TestCase):
    def test_missing_dependency_or_changed_file_rejects_checkpoint(self):
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
                with self.assertRaisesRegex(ValueError, 'exact seven'):
                    module.validate_project_sources(missing)
                (root / module.SOURCE_FILES[-1]).write_text('changed dependency')
                with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
                    module.validate_project_sources(recorded)


if __name__ == '__main__':
    unittest.main()
