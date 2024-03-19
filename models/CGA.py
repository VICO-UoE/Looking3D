
from models.load_model import MLP, get_embedder
from models.xformer import SpatialTransformer
from torch import nn
import torch, math
from einops import rearrange, reduce, repeat

class CGA_decoder(nn.Module):
    
    def __init__(self):
        super().__init__()

        d_model = 128
        n_head = 8
        d_head = 64
        n_layer = 3

        self.d_model = d_model
        self.token = nn.Embedding(1, d_model)
        self.q_pos_emb = nn.Embedding(1024, d_model)
        self.kv_spatial_pos_emb = nn.Embedding(1024, d_model)
        self.pose_to_3d_emb = nn.Linear(12, 128)

        self.crossatten = SpatialTransformer(d_model, n_head, d_head, n_layer, context_dim=2*d_model, use_linear = True, attn_type='softmax-xformers', use_checkpoint = False)

        self.bbox_layer = MLP(d_model, 256, 4, 3)
        self.cls_layer = MLP(d_model, 256, 1, 2)

        self.ff_pos_enc, ff_dim = get_embedder(10)
        self.map_to_pos_enc = nn.Linear(ff_dim, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, q, k, mask, pos_enc3d):

        fh = fw = int(math.sqrt(q.shape[1]))
        d = q.shape[-1]

        t = repeat(self.token.weight, 'n c -> b n c', b  = q.shape[0])
        qt = torch.cat([t,q],1)
        pos = self.map_to_pos_enc(rearrange(self.ff_pos_enc(pos_enc3d),  'b n h w c -> b (n h w) c'))

        o = self.crossatten(qt, context = torch.cat([k, pos], -1) , mask = mask)

        ot = o[:,0]
    
        logits = self.cls_layer(ot)
        pred_bbox = self.bbox_layer(ot)

        return logits, pred_bbox