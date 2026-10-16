"""Fixed-global-batch handoff contracts; execute only on an authorized node.

The trainer boundary test replaces CUDA transport with CPU transport, while
retaining the actual training loop, checkpoint writer and official optimizers.
It does not create a process group, launch a controller or request resources.
"""
import contextlib
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import train_subcellular_v6 as trainer
import run_subcellular_v6 as controller
from spatial_pretrain.training.subcellular_v6_optimizer import build_optimizer
from test_subcellular_v6_model import model_for
from test_subcellular_v6_optimizer import synthetic_update, training_config


def logical_minibatches(world, *, size=321, start_update=0, epoch=0, seed=79):
    """Reassemble exactly the logical order used by the real trainer loop."""
    streams = []
    for rank in range(world):
        sampler = trainer.GlobalBatchSampler(size, 64, 16, world, rank, seed)
        sampler.seek(epoch, start_update)
        stream = []
        for offset, indices in enumerate(sampler):
            update = start_update + offset // sampler.accumulation
            micro = offset % sampler.accumulation
            logical = micro * world + rank
            stream.append((update, logical, indices,
                           trainer.logical_forward_seed(seed, epoch, update, logical)))
        streams.append(stream)
    return [record for physical_micro in zip(*streams) for record in physical_micro]


class V6HandoffSamplingTests(unittest.TestCase):
    def test_two_accumulated_ranks_and_four_ranks_preserve_samples_seeds_and_tail(self):
        for size in (1, 64, 65, 321):
            for epoch in (0, 1):
                with self.subTest(size=size, epoch=epoch):
                    two = logical_minibatches(2, size=size, epoch=epoch)
                    self.assertEqual(two, logical_minibatches(4, size=size, epoch=epoch))
                    self.assertEqual(two, logical_minibatches(1, size=size, epoch=epoch))
                    self.assertEqual(len(two), 4 * ((size + 63) // 64))
                    self.assertTrue(all(len(record[2]) == 16 for record in two))
                    self.assertTrue(all(0 <= index < size for record in two for index in record[2]))

    def test_four_rank_resume_starts_at_next_unsent_global_window(self):
        full = logical_minibatches(2)
        updates = len(full) // 4
        for boundary in range(updates + 1):
            with self.subTest(boundary=boundary):
                prefix = [record for record in full if record[0] < boundary]
                suffix = logical_minibatches(4, start_update=boundary)
                self.assertEqual(prefix + suffix, full)
                self.assertTrue(all(record[0] >= boundary for record in suffix))

    def test_ddp_average_and_accumulation_give_same_logical_loss_gradients(self):
        # Simulate DDP's rank average after local no_sync accumulation. Double
        # precision tolerances acknowledge different floating-point sum orders.
        logical = logical_minibatches(4)[:4]
        gradients = {}
        for world in (2, 4):
            accumulation = 4 // world
            rank_gradients = []
            for rank in range(world):
                parameter = torch.tensor([.13, -.21], dtype=torch.float64, requires_grad=True)
                for micro in range(accumulation):
                    _, _, indices, seed = logical[micro * world + rank]
                    generator = torch.Generator().manual_seed(seed)
                    features = torch.tensor(indices, dtype=torch.float64).unsqueeze(-1) / 321
                    features = features * torch.rand(16, 2, generator=generator, dtype=torch.float64)
                    loss = (features @ parameter).sin().square().mean()
                    (loss / accumulation).backward()
                rank_gradients.append(parameter.grad)
            gradients[world] = torch.stack(rank_gradients).mean(0)
        torch.testing.assert_close(gradients[2], gradients[4], atol=1e-12, rtol=1e-12)

    def test_handoff_marker_blocks_resume_until_owner_removes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            trainer.prepare_run_directory(run, resume=False)
            checkpoint = run / 'resume.pt'
            checkpoint.write_bytes(b'preserve checkpoint')
            marker = run / 'HANDOFF_TO_4GPU_REQUESTED'
            marker.write_text('request belongs to the coordinating launcher\n')
            with self.assertRaisesRegex(RuntimeError, 'paused'):
                trainer.prepare_run_directory(run, resume=True)
            self.assertTrue(marker.exists())
            self.assertEqual(checkpoint.read_bytes(), b'preserve checkpoint')
            with self.assertRaises(FileExistsError):
                trainer.prepare_run_directory(run, resume=False)
            marker.unlink()  # Simulate only the marker owner's confirmed handoff.
            trainer.prepare_run_directory(run, resume=True)


class V6ControllerHandoffRaceTests(unittest.TestCase):
    refusal = ('[rank0]: RuntimeError: RuntimeError: '
               'Run paused for review; inspect the marker before resuming\n')

    def runner(self, directory):
        root = Path(directory)
        run = root / 'run'
        run.mkdir()
        config = dict(
            output_dir=str(run), model={'route': 'A'},
            training=dict(epochs=1, checkpoint_every=100, global_batch_size=64, batch_size=16),
            evaluation=dict(interval=100, cells_per_donor=400,
                            mask_validation_batches=16, test_donors=False))
        path = root / 'config.yaml'
        path.write_text(yaml.safe_dump(config))
        args = SimpleNamespace(
            config=path, output_dir=None, resume=True, nodes=1, gpus_per_node=2,
            training_gpus=2, deadline=time.time() + 600, eval_min_seconds=1, train_min_seconds=1)
        runner = controller.Runner(args)
        runner.control.mkdir()
        (runner.control / 'identity.json').write_text(json.dumps(runner.identity))
        (run / 'resume.pt').write_bytes(b'unchanged existing optimizer boundary')
        (run / 'progress.json').write_text(json.dumps(dict(
            step=3, total_updates=10, complete=False, model_version=6,
            architecture=controller.ARCHITECTURE, config_sha256=runner.fingerprint)))
        (run / 'schedule.json').write_text(json.dumps(dict(
            total_updates=10, model_version=6, architecture=controller.ARCHITECTURE)))
        coordination = run.with_name(run.name + '.coordination')
        coordination.mkdir()
        request = dict(schema='subcellular_v6_upgrade_request_v1', route='A', output_dir=str(run),
                       target_world_size=4, target_jobid='123456', requested_at=time.time())
        (coordination / 'upgrade_request.json').write_text(json.dumps(request))
        (run / 'HANDOFF_TO_4GPU_REQUESTED').write_text(json.dumps(request))
        runner.prepare()
        self.addCleanup(runner.lock.close)
        return runner

    def fake_process(self, text):
        def launch(*args, **kwargs):
            kwargs['stdout'].write(text)
            return SimpleNamespace(wait=lambda **wait_kwargs: 1)
        return launch

    def test_precise_prelaunch_refusal_is_a_controlled_handoff_with_original_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            log = self.refusal + 'torch.distributed.elastic.multiprocessing.errors.ChildFailedError: \n'
            with mock.patch.object(controller.subprocess, 'Popen', side_effect=self.fake_process(log)):
                self.assertFalse(runner.child(['synthetic-launcher'], 'training', 3))
            entry = json.loads((runner.control / 'commands.jsonl').read_text())
            self.assertEqual(entry['status'], 'handoff_before_trainer_start')
            self.assertEqual(entry['returncode'], 1)
            self.assertTrue((runner.run / 'HANDOFF_TO_4GPU_REQUESTED').is_file())
            self.assertEqual((runner.run / 'resume.pt').read_bytes(), b'unchanged existing optimizer boundary')

    def test_unrecognized_or_mixed_failures_are_not_silently_treated_as_handoff(self):
        for log in ('RuntimeError: CUDA out of memory\n',
                    self.refusal + '[rank1]: RuntimeError: unknown NCCL failure\n',
                    self.refusal + '[schedule] {"started": true}\n',
                    self.refusal + '[train] {"step": 4}\n',
                    self.refusal + '[checkpoint] step=4\n'):
            with self.subTest(log=log), tempfile.TemporaryDirectory() as directory:
                runner = self.runner(directory)
                with mock.patch.object(controller.subprocess, 'Popen', side_effect=self.fake_process(log)):
                    with self.assertRaises(subprocess.CalledProcessError):
                        runner.child(['synthetic-launcher'], 'training', 3)
                self.assertEqual(runner.state['status'], 'failed')

    def test_invalid_marker_or_changed_boundary_cannot_authorize_handoff(self):
        for change in ('foreign_marker', 'missing_request', 'wrong_target', 'changed_step',
                       'empty_checkpoint', 'wrong_world', 'wrong_route'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                runner = self.runner(directory)
                marker = runner.run / 'HANDOFF_TO_4GPU_REQUESTED'
                request_path = runner.run.with_name(runner.run.name + '.coordination') / 'upgrade_request.json'
                if change == 'foreign_marker':
                    marker.write_text('{}')
                elif change == 'missing_request':
                    request_path.unlink()
                elif change == 'wrong_target':
                    request = json.loads(request_path.read_text())
                    request['target_world_size'] = 2
                    request_path.write_text(json.dumps(request))
                    marker.write_text(json.dumps(request))
                elif change == 'changed_step':
                    progress_path = runner.run / 'progress.json'
                    progress = json.loads(progress_path.read_text())
                    progress['step'] = 4
                    progress_path.write_text(json.dumps(progress))
                elif change == 'empty_checkpoint':
                    (runner.run / 'resume.pt').write_bytes(b'')
                elif change == 'wrong_world':
                    runner.args.training_gpus = 4
                elif change == 'wrong_route':
                    runner.cfg['model']['route'] = 'B'
                with mock.patch.object(controller.subprocess, 'Popen',
                                       side_effect=self.fake_process(self.refusal)):
                    with self.assertRaises(subprocess.CalledProcessError):
                        runner.child(['synthetic-launcher'], 'training', 3)

    def test_controlled_handoff_returns_paused_without_claiming_progress_advanced(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.runner(directory)
            # Simulate the marker arriving only after run_foreground's stop
            # check, exactly the startup race handled by child().
            with (mock.patch.object(runner, 'prepare'),
                  mock.patch.object(runner, 'evaluate_pending', return_value=True),
                  mock.patch.object(runner, 'stopped', return_value=False),
                  mock.patch.object(controller.subprocess, 'Popen',
                                    side_effect=self.fake_process(self.refusal))):
                runner.run_foreground()
            self.assertEqual(runner.state['status'], 'paused')
            self.assertEqual(runner.state['reason'], 'handoff_before_trainer_start')
            self.assertEqual(runner.state['step'], 3)
            self.assertIsNone(runner.lock)


@unittest.skipUnless(callable(getattr(torch.optim, 'Muon', None)), 'Installed PyTorch lacks official Muon')
class V6HandoffOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(97)

    def test_world_change_restores_boundary_moments_and_full_epoch_lr_without_restart(self):
        training = training_config(epochs=1, global_batch_size=64)
        config = {'training': training}
        model = model_for('B')
        optimizer = build_optimizer(model, training)
        total = trainer.GlobalBatchSampler(321, 64, 16, world=2).updates
        for step in (1, 2):
            synthetic_update(model, optimizer, step, training, total=total)
        state = dict(
            model_version=6, architecture=trainer.ARCHITECTURE, arch_config=model.arch_config,
            config=config, config_sha256=trainer.config_hash(config), source_sha256={'source': 'frozen'},
            micro_batch_size=16, global_batch_size=64, physical_world_size=2, phase='frozen_backbone',
            step=2, epoch=0, next_update=2, complete=False, model=copy.deepcopy(model.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()), optimizer_manifest=optimizer.manifest,
        )
        restored_model = model_for('B')
        restored_optimizer = build_optimizer(restored_model, training)
        four_rank_sampler = trainer.GlobalBatchSampler(321, 64, 16, world=4)
        cursor = trainer.validate_resume(
            state, config, {'source': 'frozen'}, restored_model.arch_config,
            micro_batch=16, updates_per_epoch=four_rank_sampler.updates,
            optimizer_manifest=restored_optimizer.manifest)
        self.assertEqual(cursor, (2, 0, 2))
        four_rank_sampler.seek(cursor[1], cursor[2])
        self.assertEqual(four_rank_sampler.accumulation, 1)
        restored_model.load_state_dict(state['model'], strict=True)
        restored_optimizer.load_state_dict(state['optimizer'])
        self.assertTrue(restored_optimizer.resume_state_verified)
        self.assertEqual(restored_optimizer.steps_completed, cursor[0])
        self.assertEqual(restored_optimizer.learning_rates(), optimizer.learning_rates())
        for step in range(cursor[0] + 1, total + 1):
            synthetic_update(model, optimizer, step, training, total=total)
            synthetic_update(restored_model, restored_optimizer, step, training, total=total)
            self.assertEqual(restored_optimizer.learning_rates(), optimizer.learning_rates())
            self.assertEqual(restored_optimizer.learning_rates()['muon'],
                             training['lr_muon'] * trainer.lr_multiplier(step, total, training))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored_model.state_dict()[name], value, atol=0, rtol=0)
        changed = dict(state, source_sha256={'source': 'edited after launch'})
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            trainer.validate_resume(changed, config, {'source': 'frozen'}, model.arch_config,
                                    micro_batch=16, updates_per_epoch=total,
                                    optimizer_manifest=optimizer.manifest)

    def test_marker_during_first_microbatch_saves_only_after_complete_optimizer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            trainer.prepare_run_directory(run, resume=False)
            marker = run / 'HANDOFF_TO_4GPU_REQUESTED'

            class Dataset(torch.utils.data.Dataset):
                pad_id = 0
                lr_metadata = {'source': 'synthetic'}

                def __len__(self):
                    return 128

                def __getitem__(self, index):
                    return {'inputs': torch.tensor([index / 128, 1.])}

                def set_epoch(self, epoch):
                    self.epoch = epoch

            class Model(nn.Module):
                arch_config = {'test': 'actual trainer boundary'}

                def __init__(self):
                    super().__init__()
                    self.backbone = nn.Linear(2, 2)
                    self.backbone.provenance = {'source': 'synthetic'}
                    self.hidden = nn.Linear(2, 2)
                    self.joint_decoder = nn.Linear(2, 1)
                    self.forward_count = 0

                def set_backbone_trainable_layers(self, layers):
                    self.backbone.requires_grad_(False)

                def cuda(self):
                    return self

                def forward(self, batch):
                    self.forward_count += 1
                    if self.forward_count == 1:
                        marker.write_text('handoff requested during unfinished global batch\n')
                    return {'joint_logits': self.joint_decoder(self.hidden(batch['inputs']).tanh())}

            def collate(records, **kwargs):
                return {'inputs': torch.stack([record['inputs'] for record in records])}

            def loss_function(output, batch, config, step):
                loss = output['joint_logits'].square().mean()
                return loss, {'total': float(loss.detach()), 'masked_ce': float(loss.detach())}

            real_tensor = torch.tensor

            def cpu_tensor(*args, **kwargs):
                if kwargs.get('device') == 'cuda':
                    kwargs['device'] = 'cpu'
                return real_tensor(*args, **kwargs)

            class CPUTransport:
                # Scope transport shims to the trainer module: globally
                # replacing torch.tensor/autocast can disrupt lazy optimizer
                # imports and is unnecessary for the optimizer-boundary test.
                tensor = staticmethod(cpu_tensor)
                autocast = staticmethod(lambda *args, **kwargs: contextlib.nullcontext())
                cuda = SimpleNamespace(
                    manual_seed_all=lambda *args: None,
                    get_device_name=lambda *args: 'CPU boundary test',
                    reset_peak_memory_stats=lambda: None,
                    max_memory_allocated=lambda: 0,
                    max_memory_reserved=lambda: 0,
                )

                def __getattr__(self, name):
                    return getattr(torch, name)

            training = training_config(
                epochs=1, global_batch_size=64, batch_size=16, num_workers=0,
                grad_clip_norm=1., checkpoint_every=99, resume_every=99, log_every=99)
            config = dict(training=training, data={'seed': 79, 'center_stride': 1},
                          assets_dir='synthetic', model={}, loss={'type': 'masked_bin_cross_entropy'},
                          protocol={'bins': 'synthetic'})
            args = SimpleNamespace(batch_size=None, resume=False, initialize_only=False, run_update_limit=0)
            model = Model()
            patches = (
                mock.patch.object(trainer, 'V6Dataset', return_value=Dataset()),
                mock.patch.object(trainer, 'SubcellularV6', return_value=model),
                mock.patch.object(trainer, 'collate_v6', side_effect=collate),
                mock.patch.object(trainer, 'losses', side_effect=loss_function),
                mock.patch.object(trainer, 'source_hash', return_value={'source': 'frozen'}),
                mock.patch.object(trainer, 'torch', CPUTransport()),
                mock.patch.object(torch.Tensor, 'cuda', lambda self, **kwargs: self),
                mock.patch.dict('os.environ', {'V6_TRAIN_DEADLINE': 'inf', 'LOCAL_RANK': '0'}),
                mock.patch('builtins.print'),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                trainer._train(config, args, run, rank=0, world=1)
            self.assertEqual(model.forward_count, 4)
            saved = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
            self.assertEqual((saved['step'], saved['epoch'], saved['next_update']), (1, 0, 1))
            self.assertEqual(saved['optimizer']['steps_completed'], 1)
            self.assertFalse(saved['complete'])
            self.assertTrue((run / 'checkpoints/step_00000001.pt').is_file())
            self.assertFalse((run / 'TRAIN_COMPLETE').exists())
            self.assertTrue(marker.exists())
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
