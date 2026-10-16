"""CPU regressions for rank-local DDP resume without replaying dataset reads."""
import copy
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_subcellular_v2 import ResumeBatchSampler, accumulation_window


class IndexedDataset:
    """Like V2Dataset, randomness belongs to index/epoch, not worker/global RNG."""
    def __init__(self, length=67, epoch=0):
        self.length = length
        self.epoch = epoch
        self.read_indices = []

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.read_indices.append(index)
        rng = np.random.default_rng(np.random.SeedSequence([42, self.epoch, index]))
        return {'index': index, 'x': torch.from_numpy(rng.normal(size=3)),
                'y': torch.tensor(index / self.length, dtype=torch.float64)}


def rank_sampler(dataset, rank=0, epoch=0):
    sampler = DistributedSampler(dataset, num_replicas=4, rank=rank, seed=42,
                                 shuffle=True, drop_last=False)
    sampler.set_epoch(epoch)
    return sampler


def cpu_rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_cpu_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class DDPResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_seek_matches_original_rank_batches_without_reading_old_items(self):
        for epoch in (0, 3):
            for rank in range(4):
                for start_micro in (0, 1, 4, 5):
                    with self.subTest(epoch=epoch, rank=rank, start=start_micro):
                        dataset = IndexedDataset(epoch=epoch)
                        sampler = rank_sampler(dataset, rank, epoch)
                        reference = list(BatchSampler(sampler, batch_size=4, drop_last=False))
                        seeking = ResumeBatchSampler(sampler, 4)
                        seeking.seek(start_micro)
                        loader = DataLoader(dataset, batch_sampler=seeking, num_workers=0)
                        actual = [(micro, batch['index'].tolist()) for micro, batch in
                                  enumerate(loader, start=start_micro)]
                        expected = list(enumerate(reference[start_micro:], start=start_micro))
                        self.assertEqual(actual, expected)
                        self.assertEqual(dataset.read_indices,
                                         [index for batch in reference[start_micro:] for index in batch])
                        self.assertEqual(seeking.total_batches, 5)
                        self.assertEqual(len(loader), 5 - start_micro)
                        if actual:
                            self.assertEqual(len(actual[-1][1]), 1)

    def test_current_epoch_schedule_and_resume_strided_groups_remain_unchanged(self):
        dataset = range(657203)
        permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
        permutation += permutation[:1]  # original DistributedSampler pads to a multiple of four
        for rank in range(4):
            seeking = ResumeBatchSampler(rank_sampler(dataset, rank), 16)
            seeking.seek(3652)
            batches = list(seeking)
            self.assertEqual(batches[0], permutation[3652 * 64 + rank:3653 * 64:4])
            self.assertEqual(batches[-1], permutation[10268 * 64 + rank::4])
            self.assertEqual(len(batches[-1]), 13)
            self.assertEqual(seeking.total_batches, 10269)
            self.assertEqual(len(seeking), 6617)
            self.assertEqual(math.ceil(seeking.total_batches / 1), 10269)
            self.assertEqual(math.ceil(seeking.total_batches / 2), 5135)

    def test_seek_reuses_new_epoch_shuffle_and_rejects_invalid_position(self):
        dataset = IndexedDataset()
        sampler = rank_sampler(dataset)
        seeking = ResumeBatchSampler(sampler, 4)
        seeking.seek(2)
        self.assertEqual(list(seeking), list(BatchSampler(sampler, 4, False))[2:])
        sampler.set_epoch(1)
        seeking.seek(0)
        self.assertEqual(list(seeking), list(BatchSampler(sampler, 4, False)))
        self.assertEqual(len(seeking), 5)
        for invalid in (-1, 6):
            with self.assertRaisesRegex(ValueError, 'outside'):
                seeking.seek(invalid)

    def test_absolute_accumulation_windows_preserve_final_partial_update(self):
        # A resumed epoch has only three batches left, but positions remain 2,3,4.
        actual = [accumulation_window(micro, 5, 2) for micro in range(2, 5)]
        self.assertEqual(actual, [(2, False), (2, True), (1, True)])
        self.assertEqual([micro + 1 for micro in range(2, 5)
                          if accumulation_window(micro, 5, 2)[1]], [4, 5])

    def test_dropout_optimizer_and_rng_match_uninterrupted_and_old_skip_resume(self):
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)
        model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(.3), nn.Linear(5, 1)).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
        dataset = IndexedDataset(epoch=3)
        full_loader = DataLoader(dataset, batch_size=4, sampler=rank_sampler(dataset, epoch=3))
        all_batches = list(full_loader)  # one normal DataLoader worker-base-seed draw

        def loss_for(module, batch):
            scale = 1 + .01 * (random.random() + np.random.rand())
            return (module(batch['x']).squeeze(-1) - batch['y']).square().mean() * scale

        # Reference performs one update per two full-epoch batches, then one tail.
        def reference_update(batches):
            optimizer.zero_grad(set_to_none=True)
            for batch in batches:
                (loss_for(model, batch) / len(batches)).backward()
            optimizer.step()

        reference_update(all_batches[:2])
        checkpoint_model = copy.deepcopy(model.state_dict())
        checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())
        pristine_checkpoint_optimizer = copy.deepcopy(checkpoint_optimizer)
        checkpoint_rng = cpu_rng_state()
        reference_update(all_batches[2:4])
        reference_update(all_batches[4:])
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_rng = cpu_rng_state()

        for mode in ('old_skip', 'direct_seek'):
            with self.subTest(mode=mode):
                replica = copy.deepcopy(model)
                replica.load_state_dict(checkpoint_model)
                resumed_optimizer = torch.optim.AdamW(replica.parameters(), lr=.002)
                # On the same device/dtype, optimizer load_state_dict may retain
                # the supplied moment tensors. Each scenario needs its own copy
                # so the old-skip update cannot modify the direct-seek starting
                # checkpoint (real independent torch.load calls also do this).
                resumed_optimizer.load_state_dict(copy.deepcopy(checkpoint_optimizer))
                resumed_optimizer.zero_grad(set_to_none=True)
                # Model/worker startup before restoration may consume arbitrary RNG.
                random.seed(101); np.random.seed(102); torch.manual_seed(103)
                restored_dataset = IndexedDataset(epoch=3)
                sampler = rank_sampler(restored_dataset, epoch=3)
                if mode == 'direct_seek':
                    seeking = ResumeBatchSampler(sampler, 4)
                    seeking.seek(2)
                    loader = DataLoader(restored_dataset, batch_sampler=seeking)
                    start = 2
                else:
                    loader = DataLoader(restored_dataset, batch_size=4, sampler=sampler)
                    start = 0
                restored = False
                update_positions = []
                for micro, batch in enumerate(loader, start=start):
                    if micro < 2:
                        continue
                    if not restored:
                        restore_cpu_rng(checkpoint_rng)
                        restored = True
                    size, boundary = accumulation_window(micro, len(all_batches), 2)
                    (loss_for(replica, batch) / size).backward()
                    if boundary:
                        resumed_optimizer.step()
                        resumed_optimizer.zero_grad(set_to_none=True)
                        update_positions.append(micro + 1)
                self.assertEqual(update_positions, [4, 5])
                for key, tensor in replica.state_dict().items():
                    torch.testing.assert_close(tensor, expected_model[key], atol=0, rtol=0)
                for key, values in resumed_optimizer.state_dict()['state'].items():
                    for name, tensor in values.items():
                        torch.testing.assert_close(tensor, expected_optimizer['state'][key][name], atol=0, rtol=0)
                for key, values in checkpoint_optimizer['state'].items():
                    for name, tensor in values.items():
                        torch.testing.assert_close(tensor,
                            pristine_checkpoint_optimizer['state'][key][name], atol=0, rtol=0)
                actual_rng = cpu_rng_state()
                self.assertEqual(actual_rng[0], expected_rng[0])
                np.testing.assert_array_equal(actual_rng[1][1], expected_rng[1][1])
                self.assertEqual(actual_rng[1][2:], expected_rng[1][2:])
                torch.testing.assert_close(actual_rng[2], expected_rng[2], atol=0, rtol=0)
                expected_read_count = 9 if mode == 'direct_seek' else 17
                self.assertEqual(len(restored_dataset.read_indices), expected_read_count)


if __name__ == '__main__':
    unittest.main()
