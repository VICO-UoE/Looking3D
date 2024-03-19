import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math

from models.VLFA.resnet import resnet18

class VLFA_Projector(nn.Module):
    def __init__(self, in_dim, latent_dim, proj_dim):
        super(VLFA_Projector, self).__init__()
        
        self.mlp = nn.Sequential(
                nn.Conv2d(in_dim, latent_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                nn.Conv2d(latent_dim, proj_dim, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        return torch.nn.functional.normalize(self.mlp(x),dim=1)


class VLFA_Contrast(nn.Module):

    def __init__(self, inputSize, T=0.07, negative_source=None,
            n_pos_pts=None, n_obj=None):
        super(VLFA_Contrast, self).__init__()

        self.inputSize = inputSize

        
        self.T = T
        self.negative_source = negative_source

        ## pre-initializing indexing variables

        M_size = n_obj*n_pos_pts
        self.pos_idx = torch.eye(M_size, M_size).bool()
        M_size = n_obj*n_pos_pts

        # for 3 obj 2 pts block diag would look like this
        # 1 1 0 0 0 0
        # 1 1 0 0 0 0
        # 0 0 1 1 0 0
        # 0 0 1 1 0 0
        # 0 0 0 0 1 1
        # 0 0 0 0 1 1
        block_diag = torch.block_diag(*([torch.ones(n_pos_pts, n_pos_pts)]*n_obj))

        # negatives from the same object in the 2nd view
        idx_mat_1 = (block_diag - torch.eye(M_size, M_size))
        # 0 1 0 0 0 0
        # 1 0 0 0 0 0
        # 0 0 0 1 0 0
        # 0 0 1 0 0 0
        # 0 0 0 0 0 1
        # 0 0 0 0 1 0
        # for each row, what are the columns where we have 1
        # [[1],[0],[4],[3],[5],[4]]
        self.gather_idx_1 = idx_mat_1.nonzero()[:,1].reshape(-1,n_pos_pts-1).cuda()

         # negatives from the same object in the 2nd view
        idx_mat_2 = 1 - block_diag
        # 0 0 1 1 1 1
        # 0 0 1 1 1 1
        # 1 1 0 0 1 1
        # 1 1 0 0 1 1
        # 1 1 1 1 0 0
        # 1 1 1 1 0 0
        # for each row, what are the columns where we have 1
        # [[2,3,4,5],[2,3,4,5],[0,1,4,5],[0,1,4,5] ...
        self.gather_idx_2 = idx_mat_2.nonzero()[:,1].reshape(-1, M_size-n_pos_pts).cuda()

        assert len(self.negative_source) > 0, "can't train without sampling any negatives"

    def extract_local_features(self, q, k, uv_q, uv_k):
        '''
        q: B x 128 x 56 x 56 input features 1
        k: B x 128 x 56 x 56 input features 2
        uv_q: N_pts x 2 pixel coordinates to extract features from q
        uv_k: N_pts x 2 pixel coordinates to extract features from k

        outputs extracted features from q and k of shape B, N_pts, C
        '''

        B, C, fH, fW = q.shape

        # positive features from the foreground in 2nd view
        q = q[torch.arange(B)[:,None], :, uv_q[:,:,0], uv_q[:,:,1]]
        k = k[torch.arange(B)[:,None], :, uv_k[:,:,0], uv_k[:,:,1]]

        return q, k

    def get_pos_neg_l(self, M):

        l_pos = M[self.pos_idx]
        l_neg_2nd_view = M.gather(1, self.gather_idx_1)
        l_neg_other_obj = M.gather(1, self.gather_idx_2)

        return l_pos, l_neg_2nd_view, l_neg_other_obj

    def forward(self, q, k):
        k = k.detach()

        feature_similarity_matrix = torch.mm(q, k.T)

        l_pos, l_neg_2nd_view, l_neg_other_obj = self.get_pos_neg_l(feature_similarity_matrix)

        ### return 0 if not used
        for_logging = {
                "local_pos_l":l_pos.detach().mean().item(),
                "2nd_view_neg_l":0,
                "other_obj_neg_l":0,
            }

        # pos logit
        out = l_pos.view(len(l_pos), 1)

        if "2nd_view" in self.negative_source:
            out = torch.cat([out, l_neg_2nd_view], dim=1)
            for_logging["2nd_view_neg_l"] = l_neg_2nd_view.detach().mean().item()

        if "other_obj" in self.negative_source:
            out = torch.cat([out, l_neg_other_obj], dim=1)
            for_logging["other_obj_neg_l"] = l_neg_other_obj.detach().mean().item()

        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out, for_logging

class VLFA_NetworkCNN(nn.Module):
    def __init__(self, fpn_dim):
        super(VLFA_NetworkCNN, self).__init__()

        # FPN dimension
        D = fpn_dim
        self.backbone = resnet18()

        self.fpn = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512], D)
        self.panoptic_1 = self._make_upsample_block(D, 0, 3)
        self.panoptic_2 = self._make_upsample_block(D, 16, 1)
        self.panoptic_3 = self._make_upsample_block(D, 8, 2)
        #self.panoptic_4 = self._make_upsample_block(D, 8, 3)

        self.conv1x1_mask = nn.Conv2d(D, 1,kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        backbone_out = self.backbone(x)

        # plugging features into FPN
        feats = {
                "f1":backbone_out["f5"],
                "f2":backbone_out["f6"],
                "f3":backbone_out["f7"],
                "f4":backbone_out["f8"]
            }

        _, f1, f2, f3 = self.fpn(feats).values()

        # panoptifc FPN upsampling
        f1 = self.panoptic_1(f1)
        f2 = self.panoptic_2(f2)
        f3 = self.panoptic_3(f3)

        # element-wise sum + 1x1 conv "projection"
        f = f1 + f2 + f3

        output = {}
        output['local_feat_pre_proj'] = f

        # mask prediction
        mask = self.conv1x1_mask(f)
        mask = torch.sigmoid(mask)

        output['mask'] = mask.squeeze()

        return output

    def _make_upsample_block(self, dim, in_size, blocks):
        layers = []

        for _ in range(blocks):
            conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
            norm = nn.BatchNorm2d(dim)
            activation = nn.ReLU(inplace=True)

            if in_size != 0:
                upsample = Interpolate(in_size*2, mode='bilinear')
                in_size*=2
                layers.extend([conv, norm, activation, upsample])
            else:
                layers.extend([conv, norm, activation])

        return nn.Sequential(*layers)

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

if __name__ == "__main__":

    model = VLFA_NetworkCNN(128)

    input = torch.randn(32,3,224,224)
    output = model(input)

    import pdb; pdb.set_trace()
