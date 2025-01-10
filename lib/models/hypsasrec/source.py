import torch
import torch.nn as nn
from geoopt import PoincareBallExact

from lib.models.sasrec.source import SASRec


class HypSASRec(SASRec):
    def __init__(self, config, item_num):
        super().__init__(config, item_num)

        train_curv = config.get('train_curv', False)
        print(f"train_curv: {train_curv}")
        if train_curv:
            self.c = nn.Parameter(torch.as_tensor(1.0))
            self.c.requires_grad = True
        else:
            self.c = torch.as_tensor(1.0)
        self.geom = config['geom']
        if self.geom == "ball":
            ball = PoincareBallExact(c=self.c)
            self.manifold = ball
            self.scaler = nn.Parameter(
                torch.zeros(self.item_emb.num_embeddings),
                requires_grad = config['bias']
            )
        else:
            raise NotImplementedError

    def forward(self, log_seqs):
        log_feats = self.log2feats(log_seqs) # batch x seq.length x emb.dim
        hyp_feats = self.manifold.expmap0(log_feats)
        logits = self.poincare_hyperplane(hyp_feats) # batch x seq.length x num.items
        return logits

    def poincare_hyperplane(self, hyp_feats):
        weight = self.item_emb.weight
        w_norm = weight.norm(dim=-1)
        w_unit = weight / w_norm.unsqueeze(-1)
        return unidirectional_poincare_mlr(hyp_feats, w_norm, w_unit, self.scaler, self.c)


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    denom_safe = torch.clamp_min(1. - cx2, 1e-15)

    return 2 * z_norm / rc * torch.arcsinh(
        (2. * torch.matmul(rcx, z_unit.T) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / denom_safe
    )