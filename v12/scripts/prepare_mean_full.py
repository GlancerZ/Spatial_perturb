"""Select a measured large batch, then launch full ready48 Muon training."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spatial_pretrain.config import load_config

cfg=load_config(ROOT/'configs/xenium_compartment_ready48.yaml')
cfg.experiment.name='cytoplasm_mean_full_muon_v1'
cfg.experiment.output_dir=str(ROOT/'outputs'/cfg.experiment.name)
cfg.data.cache_dirs=[str(Path('/scratch/glancerz/spatial-pretrain-v3/cache/xenium_compartments')/Path(p).name)
                     for p in cfg.data.cache_dirs]
cfg.data.cache_dir=cfg.data.cache_dirs[0]
cfg.data.max_cells=0
cfg.data.subgraph_size=256
cfg.data.knn_k=cfg.data.nearest_k=cfg.data.candidate_k=255
cfg.data.batch_size=8
cfg.data.num_workers=4
cfg.model.cell_attention_mode='cytoplasm_mean'
cfg.model.cell_heads=16
cfg.optimizer.name='muon'
cfg.optimizer.muon_adjust_lr_fn='match_rms_adamw'
cfg.optimizer.lr=cfg.optimizer.muon_lr=2e-5
cfg.training.epochs=1
cfg.training.grad_accum_steps=1
cfg.training.max_train_steps=0
cfg.training.run_update_limit=0
cfg.training.warmup_ratio=.05
cfg.training.min_lr_ratio=.1
cfg.training.max_val_batches=100
cfg.training.val_every=100
cfg.training.log_every=10
cfg.training.checkpoint_every=25
cfg.training.save_initial_checkpoint=True
cfg.training.auto_resume=True
cfg.wandb.enabled=False
path=ROOT/'configs/cytoplasm_mean_full_muon_v1.yaml'
if Path(cfg.experiment.output_dir,'resume.pt').exists():
    raise RuntimeError('Full run already has a checkpoint; resume its original config directly')
for source,cache in zip(cfg.data.h5ad_paths,cfg.data.cache_dirs):
    assert Path(source).exists(),source
    meta=json.loads((Path(cache)/'metadata.json').read_text())
    assert meta['max_cells']==0,(cache,meta['max_cells'])
print('[full-data] 48 full caches available; no cell cap, no train-step cap',flush=True)
path.write_text(yaml.safe_dump(asdict(cfg),sort_keys=False))
passed=[]
for batch in (8,16,32,64):
    log=ROOT/'logs'/f'mean_full_batch_{batch}.log'
    with log.open('w') as stream:
        result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            'scripts/benchmark_compartment_batch.py','--batch',str(batch),'--config',str(path),
            '--output-prefix','mean_full_batch'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        if 'out of memory' not in log.read_text().lower():
            raise RuntimeError(f'Non-capacity error in {log}')
        print(f'[capacity] batch={batch} OOM; retain smaller candidates',flush=True)
        break
    stats=json.loads((ROOT/'logs'/f'mean_full_batch_{batch}.json').read_text())
    stats['subgraphs_per_second']=batch*2/stats['seconds_per_update']
    print('[capacity] '+json.dumps(stats),flush=True)
    if stats['peak_reserved_gib']>.88*stats['total_gib']:
        break
    passed.append(stats)
if not passed:
    raise RuntimeError('No large batch passed with memory headroom; inspect before training')
selected=max(passed,key=lambda s:s['subgraphs_per_second'])
cfg.data.batch_size=selected['batch_per_gpu']
path.write_text(yaml.safe_dump(asdict(cfg),sort_keys=False))
(ROOT/'logs/mean_full_batch_selection.json').write_text(json.dumps(dict(selected=selected,candidates=passed),indent=2))
print('[selected] '+json.dumps(selected),flush=True)
