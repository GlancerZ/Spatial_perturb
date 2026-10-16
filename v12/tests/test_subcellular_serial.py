"""Small CPU tests for the single-device logical-four-rank continuation bridge."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_subcellular_serial import (SerialRankBatchSampler, accumulate_virtual_ranks,
    capture_rng, restore_rng, make_resume_payload, validate_resume)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Dropout(.25))
        self.backbone.provenance = {'backbone_kind': 'toy'}
        self.head = nn.Linear(8, 2)

    def forward(self, batch):
        return self.head(self.backbone(batch['x']))


def objective(output, batch):
    multiplier = 1 + .01 * (random.random() + np.random.rand())
    loss = (output - batch['y']).square().mean() * multiplier
    return loss, {'total': float(loss.detach())}


def rng_streams():
    streams = []
    for rank in range(4):
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        streams.append(capture_rng(cuda=False))
    return streams


class SerialBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def test_four_rank_mean_gradients_match_serial_accumulation_with_dropout(self):
        model = Toy().double().train()
        batches = [dict(x=torch.arange(12, dtype=torch.float64).reshape(3, 4) / 7 + rank,
                        y=torch.full((3, 2), rank / 5, dtype=torch.float64)) for rank in range(4)]
        initial = rng_streams()
        independent_gradients, expected_rngs = [], []
        for rank in range(4):
            replica = copy.deepcopy(model)
            restore_rng(initial[rank], cuda=False)
            loss, _ = objective(replica(batches[rank]), batches[rank])
            loss.backward()
            independent_gradients.append([parameter.grad.clone() for parameter in replica.parameters()])
            expected_rngs.append(capture_rng(cuda=False))
        actual_rngs = copy.deepcopy(initial)
        accumulate_virtual_ranks(model, iter(batches), actual_rngs, objective, cuda=False)
        for index, parameter in enumerate(model.parameters()):
            expected = torch.stack([gradients[index] for gradients in independent_gradients]).mean(0)
            torch.testing.assert_close(parameter.grad, expected, atol=1e-12, rtol=1e-12)
        for expected, actual in zip(expected_rngs, actual_rngs):
            self.assertEqual(expected['python'], actual['python'])
            np.testing.assert_array_equal(expected['numpy'][1], actual['numpy'][1])
            self.assertEqual(expected['numpy'][2:], actual['numpy'][2:])
            torch.testing.assert_close(expected['torch'], actual['torch'], atol=0, rtol=0)

    def test_sampler_order_padding_resume_and_tail_match_original_rank_loaders(self):
        dataset = list(range(67))
        reference = []
        for rank in range(4):
            sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, shuffle=True,
                                         seed=42, drop_last=False)
            sampler.set_epoch(3)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=False,
                                generator=torch.Generator().manual_seed(1234))
            reference.append([batch.tolist() for batch in loader])
        expected = [reference[rank][micro] for micro in range(len(reference[0])) for rank in range(4)]
        full = SerialRankBatchSampler(dataset, 4, 4, 42, 3)
        self.assertEqual(list(full), expected)
        self.assertEqual(full.updates_per_epoch, 5)
        self.assertEqual(sum(map(len, expected)), 68)  # one DistributedSampler padding repeat
        self.assertEqual([len(batch) for batch in expected[-4:]], [1, 1, 1, 1])
        resumed = SerialRankBatchSampler(dataset, 4, 4, 42, 3, start_micro=2)
        self.assertEqual(list(resumed), expected[8:])
        self.assertEqual(len(resumed), 12)

    def test_original_resume_2255_uses_strided_rank_groups(self):
        sampler = SerialRankBatchSampler(list(range(657203)), 16, 4, 42, 0, start_micro=2255)
        next_groups = list(iter(sampler))[:4]
        for rank in range(4):
            self.assertEqual(next_groups[rank], sampler.rank_indices[rank][2255 * 16:2256 * 16])
        self.assertEqual(sampler.updates_per_epoch, 10269)
        self.assertEqual([len(indices[-13:]) for indices in sampler.rank_indices], [13] * 4)

    def test_checkpoint_keeps_logical_world_four_and_all_rank_rng_states(self):
        model = Toy()
        optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 1e-5, 'base_lr': 1e-5},
                                      {'params': model.head.parameters(), 'lr': 1e-4, 'base_lr': 1e-4}])
        model({'x': torch.ones(2, 4)}).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cfg = {'training': dict(expected_world_size=4, batch_size=16, grad_accum_steps=1,
                                unfreeze_step=100, checkpoint_every=100, resume_every=25)}
        fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        rank_states = rng_streams()
        for rank, state in enumerate(rank_states):
            state['cuda'] = torch.tensor([rank, 1, 2], dtype=torch.uint8)  # schema-only CPU fixture
        payload = make_resume_payload(model, optimizer, cfg, fingerprint, 2256, 0, 2256, rank_states)
        self.assertEqual(validate_resume(cfg, payload), fingerprint)
        self.assertEqual(payload['world_size'], 4)  # original DDP trainer's compatibility check
        self.assertEqual(payload['logical_world_size'], 4)
        self.assertEqual(payload['physical_world_size'], 1)
        self.assertEqual(payload['execution_mode'], 'serial_logical_ranks')
        self.assertEqual(payload['micro_step'], 2256)
        replica = Toy()
        replica.load_state_dict(payload['model'], strict=True)
        restored_optimizer = torch.optim.AdamW([
            {'params': replica.backbone.parameters()}, {'params': replica.head.parameters()}])
        restored_optimizer.load_state_dict(payload['optimizer'])
        self.assertEqual(restored_optimizer.param_groups[0]['base_lr'], 1e-5)
        original_states = list(optimizer.state.values())
        restored_states = list(restored_optimizer.state.values())
        self.assertGreater(len(original_states), 0)
        for original, restored in zip(original_states, restored_states):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(original[key], restored[key], atol=0, rtol=0)
        for rank in range(4):
            torch.testing.assert_close(payload['rng_by_rank'][rank]['cuda'], rank_states[rank]['cuda'])
        broken = copy.deepcopy(payload)
        broken['step'] = 1
        with self.assertRaisesRegex(ValueError, 'fully-unfrozen'):
            validate_resume(cfg, broken)
        cfg['training']['grad_accum_steps'] = 2
        with self.assertRaisesRegex(ValueError, 'accumulation=1'):
            validate_resume(cfg, payload)


if __name__ == '__main__':
    unittest.main()
