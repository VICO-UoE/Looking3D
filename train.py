# Copyright (c) 2024 Ankan Bhunia
# This code is licensed under MIT license (see LICENSE file for details)

import os
import warnings

warnings.filterwarnings("ignore")

import time, cv2, torch, wandb, sys,copy, math
import torch.distributed as dist
from torch import nn, optim
from tqdm import tqdm
import numpy as np
from data.dataset_e2e import get_dataloaders
import time, kornia, torchvision
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score
from bounding_box import bounding_box as bb
from models.VLFA.moco_func_utils import (
    moment_update,
    NCESoftmaxLoss,
    batch_shuffle_ddp,
    batch_unshuffle_ddp,
)
from models.VLFA.vlfa_block  import VLFA_Contrast, VLFA_NetworkCNN, VLFA_Projector 
from models.CGA import CGA_decoder
from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.visualize import obtain_vis_maps
from einops import rearrange, reduce, repeat 
import torch.nn.functional as F

#os.environ["WANDB_API_KEY"] = "XXXX" ## enter your wandb token here.
os.environ["WANDB_MODE"] = "offline"

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
            backend="gloo",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_main_process():
    try:
        if dist.get_rank()==0:
            return True
        else:
            return False
    except:
        return True


def build_network(batch_size, device, distributed, resume_ckpt, local_rank):

    """
    Parameters:
        batch_size (int): Batch size for data processing.
        device (str): Device to run the model ('cuda' or 'cpu').
        distributed (bool): Flag indicating whether to use distributed training.
        resume_ckpt (str): Path to a checkpoint file for resuming training.
        local_rank (int): Local rank of the current process in a distributed setting.

    Returns:
    - network_components (list): A list containing the constructed network components in the following order: encoder, encoder_ema, projector, projector_ema, attention_decoder, and local_contrast.
    """

    assert batch_size>1, "Batchsize needs to be greater than 1 to load the VLFA module."
    
    encoder = VLFA_NetworkCNN(
            128)
    projector = VLFA_Projector(
            128,
            128*8,
            256)
    encoder_ema = VLFA_NetworkCNN(
            128)
    projector_ema = VLFA_Projector(
            128,
            128*8,
            256)

    attention_decoder = CGA_decoder()

    moment_update(encoder, encoder_ema, 0)
    moment_update(projector, projector_ema, 0)

    for name, p in encoder_ema.named_parameters():
        p.requires_grad = False

    for name, p in projector_ema.named_parameters():
        p.requires_grad = False
        
    encoder = encoder.to(device)
    encoder_ema = encoder_ema.to(device)
    projector = projector.to(device)
    projector_ema = projector_ema.to(device)
    attention_decoder = attention_decoder.to(device)

    local_contrast = VLFA_Contrast(
            256, 
            T=0.1, 
            negative_source=["2nd_view", "other_obj"],
            n_pos_pts=32, 
            n_obj=batch_size
        )

    if distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[local_rank],find_unused_parameters=True
        )
        encoder_without_ddp = encoder.module

        projector = nn.parallel.DistributedDataParallel(
            projector,
            device_ids=[local_rank],find_unused_parameters=True
        )
        projector_without_ddp = projector.module

        attention_decoder = nn.parallel.DistributedDataParallel(
            attention_decoder,
            device_ids=[local_rank],find_unused_parameters=True
        )
        attention_decoder_without_ddp = attention_decoder.module

    else:   
        pass
        #print ('Single-GPU not supported.')

    if resume_ckpt is not None:

        ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)

        if distributed:
            encoder.module.load_state_dict(ckpt["encoder"], strict=False)
            projector.module.load_state_dict(ckpt["projector"], strict=False)
            attention_decoder.module.load_state_dict(ckpt["attention_decoder"], strict=False)

        else:
            encoder.load_state_dict(ckpt["encoder"])
            projector.load_state_dict(ckpt["projector"])
            attention_decoder.load_state_dict(ckpt["attention_decoder"])

        projector_ema.load_state_dict(ckpt["projector_ema"])
        encoder_ema.load_state_dict(ckpt["encoder_ema"])

    
        if is_main_process():  print ('model loaded successfully')

    return [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast]


def calculate_bbox_accuracy(output, labels):

    """
    This function calculates IOU accuracy
    """

    gt_bbox = labels[1].cuda()
    pr_bbox = output['pred_bbox'].sigmoid()

    ious = bbox_iou(pr_bbox, gt_bbox, x1y1x2y2=False)
    ious = [iou.item() for iou, lab, pr_lab in zip(ious, labels[0], output['pred_label']) if (lab == 1 and pr_lab == 1)]
    bbox_accu = (np.array(ious)>0.5)
    mean_iou = ious

    return bbox_accu, mean_iou

def calculate_binary_loss(logits, labels):

    """
    Binary cross entropy loss
    """

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loss = loss_fn(logits, labels.unsqueeze(-1))
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

def _create_mask(im, size = 32):

    """
    Obtain a binary mask of a given Image tensor
    """

    im = (torchvision.transforms.functional.rgb_to_grayscale(im)<0.8).float()
    im = kornia.morphology.dilation(im, torch.ones(3,3).cuda())
    im = kornia.morphology.dilation(im, torch.ones(3,3).cuda())
    im = torch.nn.functional.interpolate(im, size, mode='bilinear').detach()[:,0]

    return im
    
def _generate_pseudo_labels(image1, image2, pxy_v1, encoder, projector):

    """
    Description:
    This function generates pseudo labels for unsupervised learning using two input images, image1 and image2, along with their corresponding positions, pxy_v1, encoded by the encoder and projector models.

    Parameters:
    - image1 (torch.Tensor): The first input image.
    - image2 (torch.Tensor): The second input image.
    - pxy_v1 (torch.Tensor): The positions corresponding to image1.
    - encoder (torch.nn.Module): The encoder model used for feature extraction.
    - projector (torch.nn.Module): The projector model used for feature projection.

    Returns:
    - pxy_c 
    """

    with torch.no_grad():

        feats1 = encoder.module(image1)
        feats2 = encoder.module(image2)

        local_feat_grid_1 = feats1['local_feat_pre_proj']
        local_feat_grid_2 = feats2['local_feat_pre_proj']

        mask_1 = _create_mask(image1, size = 32)#output_q['mask']
        mask_2 = _create_mask(image2, size = 32)#output_q['mask']


        local_feat_grid_1 = projector(local_feat_grid_1)
        local_feat_grid_2 = projector(local_feat_grid_2)

        local_feat_grid_1 = mask_1.unsqueeze(1) * local_feat_grid_1
        local_feat_grid_2 = mask_2.unsqueeze(1) * local_feat_grid_2

        B,_, H, W = image1.shape
        B, C, fH, fW = local_feat_grid_1.shape
        assert fH == fW
        feat_dim = fH

        ratio = H / fH
        f1 = rearrange(local_feat_grid_1, 'b c fh fw -> b (fh fw) c')
        f2 = rearrange(local_feat_grid_2, 'b c fh fw -> b (fh fw) c')

        dist = torch.einsum('bic,bjc->bij', f1, f2)

        max_idx = torch.argmax(dist, dim=2)

        uv_c1 = torch.floor(pxy_v1.clip(0,256-1)/ratio).long()

        uv_c1_flatten = uv_c1[:,:,0]*fH+uv_c1[:,:,1]

        uv_c2 = torch.stack([max_idx_i[uv_c1_flatten_i] for max_idx_i, uv_c1_flatten_i in zip(max_idx, uv_c1_flatten)],0)

        pred_px_v2 = torch.stack([torch.div(uv_c2, feat_dim).int(), torch.remainder(uv_c2, feat_dim)], -1)*ratio

        return pred_px_v2.detach()


def forward_vlfa(batch, models, optimizer): #fsl->fully supervied, wsl->weakly supervised, sl-> supervied

    """

    Description:
    This function calculates the VFLA loss and updates the models.

    Parameters:
    - batch (dict): A dictionary containing input data in the following format:
        - 'img_v1': batch of view image 1: torch.Size([B, 3, 256, 256])
        - 'img_v2': batch of view image 2: torch.Size([B, 3, 256, 256]) 
        - 'img_c': batch of RGB image (query): : torch.Size([B, 3, 256, 256])
        - 'pxy_v1': 32 correpondence points in pixel space between view 1 and 2: torch.Size([N, 32, 2])
        - 'pxy_v2': 32 correpondence points in pixel space between view 2 and 1: torch.Size([N, 32, 2])
        
    - models (list): A list containing the encoder, encoder_ema, projector, projector_ema, attention_decoder, and local_contrast models.
    - optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters.

    Returns:
    - updated_model (list): A list containing the updated model parameters.
    - optimizer (torch.optim.Optimizer): The updated optimizer state.
    - loss (dict): A dictionary containing the contrastive loss.

    Raises:
    - TypeError: If the provided models are not in the expected format (list).
    - ValueError: If the mode parameter is not one of the specified values ('fsl', 'wsl', 'sl').

    """

    [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = models

    loss_fn = NCESoftmaxLoss()

    img_v1, img_v2, pxy_v1, pxy_v2, img_c = batch['img_v1'], batch['img_v2'], batch['pxy_v1'], batch['pxy_v2'], batch['img_c'] 

    B, C, H, W = img_v1.shape

    img_v1 = img_v1.cuda(non_blocking=True)
    img_v2 = img_v2.cuda(non_blocking=True)

    rand_idx = torch.randperm(B)
    
    B_ps = B//2

    img_c = img_c.cuda(non_blocking=True)
    pxy_v1 = pxy_v1.cuda(non_blocking=True)
    pxy_v2 = pxy_v2.cuda(non_blocking=True)
    pxy_c = _generate_pseudo_labels(img_v1[:B_ps], img_c[:B_ps], pxy_v1[:B_ps], encoder, projector)

    img_v2 = torch.cat([img_v2[B_ps:], img_v1[:B_ps]])[rand_idx]
    img_v1 = torch.cat([img_v1[B_ps:], img_c[:B_ps]])[rand_idx]
    
    pxy_v2 = torch.cat([pxy_v2[B_ps:], pxy_v1[:B_ps]])[rand_idx]
    pxy_v1 = torch.cat([pxy_v1[B_ps:], pxy_c])[rand_idx]

    output_q = encoder(img_v1)

    with torch.no_grad():
        # shuffle for making use of BN
        img_v2, idx_unshuffle = batch_shuffle_ddp(img_v2)
        output_k = encoder_ema(img_v2)

        # undo shuffle
        output_k = {k:batch_unshuffle_ddp(v, idx_unshuffle) for k,v in output_k.items()}

    local_feat_grid_q = output_q['local_feat_pre_proj']
    local_feat_grid_k = output_k['local_feat_pre_proj']

    _, _, fH, fW = local_feat_grid_q.shape

    ratio = H/fH
    uv_c1 = torch.floor(pxy_v1.clip(0,256-1)/ratio).long()
    uv_c2 = torch.floor(pxy_v2.clip(0,256-1)/ratio).long()

    local_feat_q, local_feat_k = (
        local_contrast.extract_local_features(
            local_feat_grid_q, 
            local_feat_grid_k,
            uv_c1,
            uv_c2)
        )

    B, n_pts, C = local_feat_q.shape
    local_feat_q = local_feat_q.view(B*n_pts, C, 1, 1)
    local_feat_k = local_feat_k.view(B*n_pts, C, 1, 1)

    local_feat_q = projector(local_feat_q).squeeze()

    with torch.no_grad():
        local_feat_k = projector_ema(local_feat_k).squeeze()

    mask_q = _create_mask(img_v1, size = 32)#output_q['mask']
    mask_k = _create_mask(img_v2, size = 32)#output_q['mask']

    # mask out projected features
    local_mask_q, local_mask_k = local_contrast.extract_local_features(
        mask_q.unsqueeze(1), mask_k.unsqueeze(1), uv_c1, uv_c2)

    local_mask_q = local_mask_q.view(B*n_pts, 1)
    local_mask_k = local_mask_k.view(B*n_pts, 1)
    
    local_feat_q = local_feat_q * local_mask_q
    local_feat_k = local_feat_k * local_mask_k

    out, sim_dct = local_contrast(local_feat_q, local_feat_k)

    loss = {'closs':loss_fn(out)}

    optimizer.zero_grad()
    loss['closs'].backward()
    optimizer.step()

    moment_update(encoder, encoder_ema, 0.999)
    moment_update(projector, projector_ema, 0.999)

    updated_model = [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast]

    return updated_model, optimizer, loss

def forward_cmt(batch, models, optimizer = None, is_train = True, topk = 100):

    """
    Description:
    This function performs a forward pass through the CMT model. It takes a batch of input data containing images, mesh data, labels, bounding boxes, and 3D positional encodings, and computes the output embeddings using the provided models. 

    Parameters:
    - batch (dict): A dictionary containing input data in the following format:
        - 'imgs': Input images: torch.Size([B, 3, H, W])
        - 'mesh': Mesh data: torch.Size([B, N, 3, H, W])
        - 'labels': Labels for the input data: torch.Size([B])
        - 'bbox': Bounding boxes: torch.Size([B, 4])
        - 'pos_enc3d': 3D positional [x,y,z] encodings at 32x32 
           latent dimension for each mv image: torch.Size([B, N, 32, 32, 3])

    - models (list): A list containing the following models in order:
        - encoder: The encoder model. 
        - encoder_ema: The exponential moving average (EMA) version of the encoder.
        - projector: The projector model. 
        - projector_ema: The EMA version of the projector.
        - attention_decoder: The attention-based decoder model. 
        - local_contrast: The local contrastive model. 
    - optimizer (optimizer object, optional): The optimizer to use for optimization. Default is None.
    - is_train (bool, optional): Flag indicating whether the model is in training mode. Default is True.
    - topk (int, optional): The top-k value for attention.

    Returns:
    If is_train is True:
    - updated_model (list): A list containing the updated model parameters.
    - optimizer (torch.optim.Optimizer): The updated optimizer state.
    - loss (dict): A dictionary containing different loss components, including binary cross-entropy loss, bbox loss, giou loss, and total loss.
    - output (dict): A dictionary containing various outputs, including predictions, predicted bounding boxes, accuracy, bbox accuracy, and mean IoU.
    """    

    output = {}

    imgs, mesh, labels, bbox, pos_enc3d = batch['query_imgs'], batch['mesh_images'], batch['labels'], batch['bbox'], batch['mesh_pos_enc3d'] 

    [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = models
    
    imgs = imgs.cuda()
    mesh = mesh.cuda()
    labels = labels.cuda()
    bbox = bbox.cuda()
    mesh_batched = rearrange(mesh, 'b n c h w -> (b n) c h w')

    B1, B2 = mesh_batched.shape[0], imgs.shape[0]
    merged_batch = torch.cat([mesh_batched, imgs], 0)
    enc_output = encoder(merged_batch)

    merged_feats = enc_output['local_feat_pre_proj']
    
    mesh_feat, query_feat = merged_feats[:B1], merged_feats[B1:]

    mesh_feat = rearrange(mesh_feat, '(b n) c h w -> b (n h w) c', b = mesh.shape[0])
    query_feat = rearrange(query_feat, 'b c h w -> b (h w) c', b = mesh.shape[0])

    if topk>0:
        with torch.no_grad():
            merged_proj_feats = projector(merged_feats)
            merged_mask = _create_mask(merged_batch, size = 32)
            mesh_proj_feats, query_proj_feats = merged_proj_feats[:B1], merged_proj_feats[B1:]
            mesh_mask, query_mask = merged_mask[:B1], merged_mask[B1:]
            mesh_proj_feats = mesh_mask.unsqueeze(1)*mesh_proj_feats
            query_proj_feats = query_mask.unsqueeze(1)*query_proj_feats
            mesh_proj_feats = rearrange(mesh_proj_feats, '(b n) c h w -> b (n h w) c', b = mesh.shape[0])
            query_proj_feats = rearrange(query_proj_feats, 'b c h w -> b (h w) c', b = mesh.shape[0])
            affinity = torch.einsum('bic,bjc->bij', query_proj_feats, mesh_proj_feats)
            mask = -torch.inf*torch.ones_like(affinity)
            index = affinity.topk(k = topk, dim = -1, largest = True)[1]
            mask.scatter_(-1,index, 0.) 
            output.update({'topk_mask':mask,'topk_dot':affinity})
            output.update({'mesh_proj_feats':mesh_proj_feats, 'query_proj_feats':query_proj_feats})
            mask = torch.cat([torch.zeros((mask.shape[0],1,mask.shape[2])).cuda(), mask],1)
            mask = repeat(mask, 'b n1 n2 -> (b nh) n1 n2', nh = 8)
    else:
        mask = None

    logits, pred_bbox = attention_decoder(query_feat, mesh_feat, mask=mask, pos_enc3d=pos_enc3d)

    output.update({'pred':logits, 'pred_bbox':pred_bbox})

    output['pred_label'] = (torch.sigmoid(output['pred'])>0.5).float()[:,0]
    output['gt_label'] = labels

    acc = ((output['pred_label']==labels).float()).mean().item()
    bbox_accu, mean_iou = calculate_bbox_accuracy(output, [labels, bbox])

    output.update({'accuracy':acc, 'bbox_accu':bbox_accu, 'mean_iou': mean_iou})

    loss = {'bceloss': calculate_binary_loss(logits, labels.float())}
    bboxloss = bbox_loss(pred_bbox.sigmoid(), bbox.float(), mask = labels.unsqueeze(-1))
    loss['loss_bbox'] = bboxloss['loss_bbox']
    loss['loss_giou'] = bboxloss['loss_giou']
    loss['loss'] = loss['bceloss'] + 5*loss['loss_bbox'] + 2*loss['loss_giou']

    if is_train:
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()
    else:
        return output

    updated_model = [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast]

    return updated_model, optimizer, loss, output

def test(args, test_dataset, models):

    """
    This function returns accuracy, confusion matrix, ROC curve, 
    and other metrics. Optionally, it also computes metrics related to bounding box predictions 
    if args.pred_box is True.

    Returns:
    - result (dict): A dictionary containing various evaluation metrics such as accuracy, ROC AUC, precision, recall, F1 score, and optionally, metrics related to bounding box predictions.
    """

    test_acc = []
    all_labels = []
    all_preds = []
    all_outputs = []
    all_ious = []
    all_bboxaccus = []


    for batch in tqdm(test_dataset):

        with torch.no_grad():
            output_dict = forward_cmt(batch, models, is_train = False, topk = args.topk)
            
        all_outputs.append(output_dict['pred'])
        all_labels.append(output_dict['gt_label'])
        all_preds.append(output_dict['pred_label'])
        test_acc.append(((output_dict['pred_label']==output_dict['gt_label']).float()).mean().item())
        
        if args.pred_box:
            bbox_accu, mean_iou = output_dict['bbox_accu'], output_dict['mean_iou']
            all_ious  +=  mean_iou
            all_bboxaccus  +=  bbox_accu.tolist()

    all_labels_cat = torch.cat(all_labels,0)
    all_preds_cat = torch.cat(all_preds,0)
    all_outputs_cat = torch.cat(all_outputs,0)

    accuracy = ((all_preds_cat==all_labels_cat).float()).mean().item()
    conf_matrix = confusion_matrix(all_labels_cat.cpu().numpy(), all_preds_cat.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels_cat.cpu(), all_outputs_cat.cpu())
    roc_auc = auc(fpr, tpr)

    precision = precision_score(all_labels_cat.cpu(), all_preds_cat.cpu())
    recall = recall_score(all_labels_cat.cpu(), all_preds_cat.cpu())
    f1 = f1_score(all_labels_cat.cpu(), all_preds_cat.cpu())

    good_accuracy = conf_matrix[0][0]/conf_matrix.sum(1)[0]
    bad_accuracy = conf_matrix[1][1]/conf_matrix.sum(1)[1]

    result = {'roc_auc': roc_auc, 'accuracy':accuracy, 'good_accuracy':good_accuracy, 
                'bad_accuracy':bad_accuracy, 'precision':precision, 'recall':recall, 'f1':f1}


    if args.pred_box:
        result['iou'] = sum(all_ious)/len(all_ious)
        result['BoxAcc'] = sum(all_bboxaccus)/len(all_bboxaccus)

    return result

def train(train_dataset, test_dataset, models, optimizer, lr_scheduler, device, wandb):

    """
    The main training loop - 
    """

    i = 0
    acc_list = []
    bbox_acc_list = []
    iou_list = []

    test_loader = iter(test_dataset)

    is_train_with_pseudo_labels = True


    log_loss_cl_bbox = {}
    log_loss_contr = {}

    for epoch in range(args.epochs):

        if is_main_process: print ('#Epoch - '+str(epoch))

        start_time = time.time()

        #train_dataset.sampler.set_epoch(epoch)     

        for batch in train_dataset:

            i = i + 1

            if len(batch['query_imgs']) != train_dataset.batch_size:
                continue
            
            if not args.no_contr_loss:
                models, optimizer, log_loss_contr = forward_vlfa(batch, models, optimizer)

            models, optimizer, log_loss_cl_bbox, output = forward_cmt(batch, models, optimizer, topk = args.topk)

            acc_list.append(output['accuracy'])
            bbox_acc_list += output['bbox_accu'].tolist()
            iou_list += output['mean_iou']

            if i%args.save_wandb_logs_every_iters == 0 and is_main_process():

                try:
                    batch_test = next(test_loader)
                except:
                    test_loader = iter(test_dataset)
                    batch_test = next(test_loader)
                
                ### uncomment the below line to visualize the correpospondence maps ###
                #vis_map, vis_map2, masks = obtain_vis_maps(batch_test, models)
                
                log_score = {'train_acc':(sum(acc_list)/len(acc_list)), 'train_bbox_acc':(sum(bbox_acc_list)/len(bbox_acc_list)), 'train_iou':(sum(iou_list)/len(iou_list)), 
                            'epoch':epoch,'steps':i}
                log_score.update(log_loss_cl_bbox)
                log_score.update(log_loss_contr)
                                
                print (f'[Epoch:{epoch}] [Step:{i}] logged info:')
                print (log_score)

                wandb.log(log_score)
                
                acc_list = []
                bbox_acc_list = []
                iou_list = []
    
        if (epoch+1)%args.save_checkpoints_every_epoch == 0 and is_main_process():

            [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = models

            if args.distributed:
                encoder_module = encoder.module
                projector_module = projector.module
                attention_decoder_module = attention_decoder.module

            else:
                encoder_module = encoder
                projector_module = projector
                attention_decoder_module = attention_decoder

            torch.save(
                {
                    "encoder": encoder_module.state_dict(),
                    "encoder_ema": encoder_ema.state_dict(),
                    "projector": projector_module.state_dict(),
                    "projector_ema": projector_ema.state_dict(),
                    "attention_decoder": attention_decoder_module.state_dict(),
                },
                args.ckpt_path + f"/model_{str(epoch).zfill(6)}.pt"
            )
            
        if is_main_process():

            print ('Epoch Time '+str(int(time.time()-start_time))+' secs')

        lr_scheduler.step()


def main(args):

    if is_main_process(): wandb.init(project="Looking3D", dir = './'+args.exp_path, name = args.exp_name,  settings = wandb.Settings(code_dir="."))

    if args.distributed: local_rank = int(os.environ['LOCAL_RANK'])

    num_mesh_images = [args.num_mesh_images, args.num_mesh_images]
    
    train_dataset, test_dataset = get_dataloaders(args, num_mesh_images = num_mesh_images)

    [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = build_network(args.batch_size, args.device, args.distributed, args.resume_ckpt, args.local_rank)

    effective_lr = args.lr
    
    params_ED = [   {"params":encoder.parameters(), "lr":effective_lr},
                    {"params":projector.parameters(), "lr":effective_lr},
                    {"params":attention_decoder.parameters(), "lr":effective_lr}
                ]

    optim_ED =  torch.optim.Adam(params_ED, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
    sched_ED = torch.optim.lr_scheduler.StepLR(optim_ED, args.lr_drop)

    optimizer = optim_ED
    lr_scheduler = sched_ED
    
    models = [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast]

    train(
        train_dataset, test_dataset, models, optimizer, lr_scheduler, args.device, wandb
    )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='CMT-final')
    parser.add_argument('--data_path', type=str, default='./brokenchairs/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--pred_box', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_drop', type=float, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_wandb_logs_every_iters', type=int, default=100)
    parser.add_argument('--save_checkpoints_every_epoch', type=int, default=1)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--num_mesh_images', type=int, default=5)
    parser.add_argument('--n_pnts', type=int, default=32)
    parser.add_argument('--no_contr_loss', action='store_true')

    args = parser.parse_args()
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print ('Experiment: '+ args.exp_name)

    if args.distributed:  init_distributed()

    args.exp_path = f'experiments/{args.exp_name}'
    args.ckpt_path = f'experiments/{args.exp_name}/checkpoints'

    if is_main_process():

        os.makedirs(args.ckpt_path, exist_ok = True)
    
        with open(f'experiments/{args.exp_name}/command', 'w') as f:
            f.write(" ".join(sys.argv[:]))

    main(args)
