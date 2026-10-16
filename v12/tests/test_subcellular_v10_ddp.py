"""Two-process CPU/Gloo test, to run inside an authorized compute allocation.

This compares the actual trainer gather function plus DDP backward with one
unsharded batch, rather than asserting a scaling formula in isolation. No GPU
memory or pretrained model assets are required.
"""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_subcellular_v10 import gather_global_output
from spatial_pretrain.training.subcellular_v10 import losses


class TinyPredictors(nn.Module):
    def __init__(self):
        super().__init__()
        self.nucleus = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 4))
        self.context = nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 4))

    def forward(self, batch):
        nucleus = F.normalize(self.nucleus(batch["nuclear_features"]), dim=-1)
        context = F.normalize(self.context(batch["context_features"]), dim=-1)
        return dict(
            nucleus_prediction=nucleus * batch["nucleus_available"][:, None],
            context_prediction=context * batch["context_available"][:, None],
            teacher=batch["teacher"].detach(),
            nucleus_available=batch["nucleus_available"],
            context_available=batch["context_available"])


def full_batch(empty_rank):
    generator = torch.Generator().manual_seed(310)
    nuclear = torch.randn(64, 5, generator=generator)
    context = torch.randn(64, 5, generator=generator)
    teacher = F.normalize(torch.randn(64, 4, generator=generator), dim=-1)
    identity = torch.stack((torch.arange(64) // 16, torch.arange(64) % 16), dim=-1)
    # A real duplicate straddles the rank boundary; it must remain a positive.
    for index in (31, 32):
        identity[index] = identity[0]
        nuclear[index], context[index], teacher[index] = nuclear[0], context[0], teacher[0]
    nucleus_available = torch.arange(64).remainder(5).ne(0)
    context_available = torch.arange(64).remainder(7).ne(0)
    if empty_rank:
        nucleus_available[:32] = False
        context_available[:32] = False
        nucleus_available[32:] = True
        context_available[32:] = True
    return dict(nuclear_features=nuclear, context_features=context, teacher=teacher,
                cell_identity=identity, nucleus_available=nucleus_available,
                context_available=context_available)


def _worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120))
    try:
        for empty_rank in (False, True):
            for kind in ("cosine", "contrastive", "relational"):
                torch.manual_seed(97)
                predictor = TinyPredictors()
                reference = deepcopy(predictor)
                wrapped = DDP(predictor, find_unused_parameters=False)
                batch = full_batch(empty_rank)
                cfg = {"loss": {"type": kind, "temperature": .2,
                                "student_temperature": .2, "teacher_temperature": .1}}
                reference_output, reference_batch = gather_global_output(reference(batch), batch, 1)
                expected_loss, expected_metrics = losses(reference_output, reference_batch, cfg)
                expected_loss.backward()
                local = {name: value[rank * 32:(rank + 1) * 32] for name, value in batch.items()}
                global_output, global_batch = gather_global_output(wrapped(local), local, world)
                actual_loss, actual_metrics = losses(global_output, global_batch, cfg)
                actual_loss.backward()
                label = f"kind={kind}, empty_rank={empty_rank}, rank={rank}"
                torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-6,
                                           msg=f"Global loss mismatch: {label}")
                for key in ("unique_cells", "nucleus_cells", "context_cells", "paired_cells"):
                    if actual_metrics[key] != expected_metrics[key]:
                        raise AssertionError(f"Mask/identity gather mismatch for {key}: {label}")
                expected_parameters = dict(reference.named_parameters())
                for name, parameter in predictor.named_parameters():
                    expected = expected_parameters[name].grad
                    if parameter.grad is None or expected is None:
                        raise AssertionError(f"Missing parameter gradient {name}: {label}")
                    torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6,
                                               msg=f"DDP/global gradient mismatch for {name}: {label}")
                # Matching gradients must produce the same actual parameter update.
                torch.optim.SGD(predictor.parameters(), lr=.03).step()
                torch.optim.SGD(reference.parameters(), lr=.03).step()
                for name, parameter in predictor.named_parameters():
                    torch.testing.assert_close(parameter, expected_parameters[name], rtol=2e-6, atol=2e-6,
                                               msg=f"DDP/global update mismatch for {name}: {label}")
                dist.barrier()
                del wrapped, predictor, reference
    finally:
        dist.destroy_process_group()


class V10DistributedGradientTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "PyTorch Gloo is required")
    def test_two_rank_gradients_match_unsharded_global_batch_for_all_objectives(self):
        with tempfile.TemporaryDirectory(prefix="v10-ddp-test-") as directory:
            rendezvous = str(Path(directory) / "gloo-rendezvous")
            mp.spawn(_worker, args=(2, rendezvous), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
