import hashlib
import json
from pathlib import Path
import torch
from spatialformer.tools.get_embeddings import manual_train_fm

root = Path('resources/spatialformer')
p = root/'single_71.ckpt'
h = hashlib.md5()
with p.open('rb') as f:
    while b := f.read(8*1024*1024): h.update(b)
assert h.hexdigest()=='fb2c33044244bf21a962e106951769f5', h.hexdigest()
pkg = root/'package/spatialformer'
config = json.loads((pkg/'config/_config_train_large_pair.json').read_text())
config['embedding_path'] = str(pkg/'spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl')
model = manual_train_fm(config, use_flash_attn=False)
ckpt = torch.load(p, map_location='cpu', weights_only=False)
print('BASELINE_CHECKPOINT', list(ckpt), flush=True)
print(model.load_state_dict(ckpt['state_dict'], strict=True), flush=True)
print('BASELINE_PARAMS', sum(p.numel() for p in model.parameters()), flush=True)
p = Path('outputs/xenium_step1500_benchmark/step1500_resume.pt')
if p.exists():
    state = torch.load(p, map_location='cpu', weights_only=False)
    assert state['step']==1500
    print('OURS_STEP', state['step'], 'GENE_KEYS', [(k,tuple(v.shape)) for k,v in state['model'].items() if 'gene_embedding' in k], flush=True)
    torch.save({k:state[k] for k in ['model','config','step','signature']}, p.parent/'model1500.pt')
    print('EXPORTED_MODEL_ONLY', flush=True)
