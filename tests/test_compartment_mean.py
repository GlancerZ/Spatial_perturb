"""Masked-mean communication must ignore order, padding and token multiplicity."""
from copy import deepcopy
import unittest
import torch
from spatial_pretrain.models.compartment import CytoplasmicMeanCellAttention, CompartmentSSGTModel


class MeanCommunicationTests(unittest.TestCase):
    def test_masked_mean_routing_and_denominator(self):
        layer = CytoplasmicMeanCellAttention(4, 1, 0).double().eval()
        with torch.no_grad():
            for p in layer.parameters():
                p.zero_()
            layer.qkv.weight[8:].copy_(torch.eye(4))
            layer.proj.weight.copy_(torch.eye(4))
        tokens = torch.full((1, 3, 4, 4), 1000., dtype=torch.double, requires_grad=True)
        with torch.no_grad():
            tokens[0, 0, 1] = 2
            tokens[0, 1, 1] = 3
            tokens[0, 1, 2] = 5
            tokens[0, 2, 1:3] = 900
        comp = torch.tensor([[[0,1,2,2], [0,1,1,2], [0,1,1,2]]])
        out, active = layer(tokens, comp, comp.ne(2), torch.tensor([[0,1,1]]))
        torch.testing.assert_close(out[0,0,1], torch.full((4,),4.,dtype=torch.double))
        torch.testing.assert_close(out[0,1,1:3], torch.full((2,4),2.,dtype=torch.double))
        self.assertTrue((out[~active] == 0).all())
        out[0,0,1].sum().backward()
        self.assertEqual(tokens.grad[comp.ne(1)].abs().sum().item(),0.)

    def test_model_permutation_invariance(self):
        torch.manual_seed(33)
        model = CompartmentSSGTModel(num_genes=10,num_value_bins=4,hidden_dim=8,num_layers=2,
            gene_heads=2,cell_heads=2,dropout=0,ffn_mult=2,gene_padding_idx=9,
            cell_attention_mode='cytoplasm_mean').double().eval()
        comp=torch.tensor([[[0,1,1,2],[0,0,1,2]]])
        batch=dict(gene_ids=torch.tensor([[[0,1,2,9],[3,4,5,9]]]),
            input_bins=torch.tensor([[[1,2,4,5],[3,1,2,5]]]),compartment_ids=comp,
            token_valid_mask=comp.ne(2),node_indices=torch.tensor([[0,1]]),cell_roles=torch.tensor([[1,0]]))
        perm=torch.tensor([[[3,1,0,2],[1,2,3,0]]])
        reordered={k:v.gather(2,perm) if v.ndim==3 else v for k,v in batch.items()}
        a=model(**batch)['mask_logits']
        b=model(**reordered)['mask_logits']
        torch.testing.assert_close(a.gather(2,perm[...,None].expand_as(a)),b)
        torch.testing.assert_close(model(**batch,return_embeddings=True)['cell_embeddings'],
            model(**reordered,return_embeddings=True)['cell_embeddings'])
        captured = []
        hook = model.final_norm.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
        embedding = model(**batch,return_embeddings=True)['cell_embeddings']
        hook.remove()
        mask = comp.eq(1)
        expected = captured[0].float().masked_fill(~mask[...,None],0.).sum(2) / mask.sum(2)[...,None]
        torch.testing.assert_close(embedding, expected)

    def test_empty_cytoplasm_is_zero_with_gradients(self):
        layer=CytoplasmicMeanCellAttention(8,2,0)
        x=torch.randn(1,2,3,8,requires_grad=True)
        comp=torch.zeros(1,2,3,dtype=torch.long)
        out,active=layer(x,comp,comp.ne(2),torch.tensor([[0,1]]))
        self.assertFalse(active.any())
        self.assertEqual(out.abs().sum().item(),0.)
        out.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_flash_matches_reference(self):
        torch.manual_seed(34)
        fast=CytoplasmicMeanCellAttention(32,4,0,require_flash=True).cuda()
        reference=deepcopy(fast);reference.require_flash=False
        x=torch.randn(1,4,5,32,device='cuda',requires_grad=True)
        y=x.detach().clone().requires_grad_()
        comp=torch.tensor([[[0,1,1,2,2],[0,0,1,2,2],[0,0,2,2,2],[0,1,1,1,2]]],device='cuda')
        nodes=torch.tensor([[0,1,2,1]],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,_=fast(x,comp,comp.ne(2),nodes)
            b,_=reference(y,comp,comp.ne(2),nodes)
        torch.testing.assert_close(a,b,atol=.004,rtol=.04)
        a.float().square().mean().backward();b.float().square().mean().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=.05)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.parameters()))


if __name__=='__main__':
    unittest.main()
