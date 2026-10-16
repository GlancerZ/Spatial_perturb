"""V5 paired-anchor acceptance, identity and same-C comparison regressions."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_subcellular_v5 as evaluator
import train_subcellular_v5 as trainer


class PairedAnchorTests(unittest.TestCase):
    def test_step_zero_requires_exact_same_dimensional_plain_reference(self):
        base = np.eye(3, dtype=np.float32)
        features = dict(base=base, joint=base.copy(), intrinsic=base.copy())
        report = evaluator.anchor_comparison(features, 0, spatial_enabled=True)
        self.assertTrue(report['step_zero_identity_verified'])
        self.assertEqual(report['residual_mean_norm'], 0.)
        features['joint'][0, 0] += 1e-5
        with self.assertRaisesRegex(ValueError, 'Step-zero'):
            evaluator.anchor_comparison(features, 0, spatial_enabled=True)

    def test_later_residual_report_and_no_spatial_arm_contract(self):
        base = np.eye(3, dtype=np.float32)
        features = dict(base=base, joint=base + .1, intrinsic=base.copy())
        report = evaluator.anchor_comparison(features, 100, spatial_enabled=True)
        self.assertFalse(report['step_zero_identity_verified'])
        self.assertGreater(report['residual_to_base_mean_norm_ratio'], 0.)
        with self.assertRaisesRegex(ValueError, 'No-spatial'):
            evaluator.anchor_comparison(features, 100, spatial_enabled=False)
        features['intrinsic'] = features['joint'].copy()
        evaluator.anchor_comparison(features, 100, spatial_enabled=False)

    def test_missing_finite_or_matching_shape_reference_fails(self):
        base = np.eye(2, dtype=np.float32)
        with self.assertRaises(ValueError):
            evaluator.anchor_comparison(dict(base=base, joint=np.zeros((2, 4)), intrinsic=base),
                                        100, spatial_enabled=True)
        joint = base.copy()
        joint[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            evaluator.anchor_comparison(dict(base=base, joint=joint, intrinsic=base),
                                        100, spatial_enabled=True)

    def test_plain_and_fused_dimensions_and_source_contract_are_shared(self):
        dimensions = evaluator.expected_dimensions(True, compartment_dim=8, backbone_dim=16)
        for name in ('base', 'joint', 'intrinsic', 'joint_neighbor_shuffle'):
            self.assertEqual(dimensions[name], 16)
        self.assertEqual(dimensions['nucleus'], 8)
        self.assertEqual(evaluator.SOURCE_FILES, trainer.SOURCE_FILES)
        self.assertIn('src/spatial_pretrain/models/subcellular_v4.py', trainer.SOURCE_FILES)
        self.assertIn('src/spatial_pretrain/data/subcellular_v5.py', trainer.SOURCE_FILES)
        with self.assertRaises(ValueError):
            evaluator.validate_checkpoint_contract(dict(model_version=4, architecture='geneset_feedback_v4'))

    def test_fixed_comparison_keeps_plain_cls_in_every_task(self):
        rows = []
        for name, value in (('base', .5), ('joint', .55), ('intrinsic', .51)):
            for task, c in (('cell_type', .1), ('region', .01)):
                rows.append(dict(embedding=name, task=task, fixed_C_result=dict(
                    C=c, val_macro_f1=value, per_donor_macro_f1={'donor': value})))
        differences = evaluator.fixed_probe_comparisons(rows)
        paired = [row for row in differences if row['left'] == 'joint' and row['right'] == 'base']
        self.assertEqual(len(paired), 2)
        for row in paired:
            self.assertAlmostEqual(row['val_macro_f1_delta'], .05)


if __name__ == '__main__':
    unittest.main()
