import inspect
import json
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F

print('torch', torch.__version__, 'Muon', hasattr(torch.optim, 'Muon'), flush=True)
if hasattr(torch.optim, 'Muon'):
    print('Muon signature', inspect.signature(torch.optim.Muon), flush=True)
for width in (16, 128, 256):
    for dropout in (0.1, 0.0):
        try:
            q = torch.randn(1, 16, 4, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 16, 255, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
                out.square().mean().backward()
            torch.cuda.synchronize()
            print(json.dumps(dict(head_dim=width, dropout=dropout, flash_forward_backward='passed')), flush=True)
        except Exception as e:
            print(json.dumps(dict(head_dim=width, dropout=dropout, error=str(e))), flush=True)
