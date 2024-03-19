from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch
from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
import torch.nn.functional as F
import numpy as np
from utils.misc import NestedTensor
from einops import rearrange, reduce, repeat


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def calculate_binary_loss(logits, labels):
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(logits, labels.unsqueeze(-1))
    weight = ((1-labels.unsqueeze(-1))*2+1).detach()
    loss = loss.mean()
    return loss

def bbox_loss(logits, labels, mask):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """
    batch_size = logits.shape[0]
    # world_size = get_world_size()
    num_boxes = batch_size

    loss_bbox = F.l1_loss(logits, labels, reduction='none')
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(logits),
        xywh2xyxy(labels)
    ))

    losses = {}
    losses['loss_bbox'] = (mask*loss_bbox).sum() / num_boxes
    losses['loss_giou'] = (mask.squeeze()*loss_giou).sum() / num_boxes

    return losses

class GradualCNNBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial, max_pooling=False):
        super(GradualCNNBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.max_pooling = max_pooling
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        x = self.convs(x)
        # To make E accept more general H*W images, we add global average pooling to 
        # resize all features to 1*1*512 before mapping to latent codes
        if self.max_pooling:
            x = F.adaptive_max_pool2d(x, 1) ##### modified
        else:
            x = F.adaptive_avg_pool2d(x, 1) ##### modified
        x = x.view(-1, self.out_c)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.model = resnet50(weights="IMAGENET1K_V1")
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input, mesh, pose, labels = None):
        logits =  self.fc2(self.fc1(self.model(input).view(input.shape[0], -1)))
        output = {'pred':logits}
        if labels is not None:
            [label, _, _] = labels
            loss = {'loss': calculate_binary_loss(logits, label.float())}
            return output, loss 
        return output

