import cv2
from bounding_box import bounding_box as bb
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from einops import rearrange, reduce, repeat
import math
from utils.box_utils import xywh2xyxy

def obtain_vis_maps(batch, models):

    [encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = models
    
    def viz_local_heatmaps(image1, image2, uv_c1, uv_c2, f1, f2):

        f1 = f1.squeeze()
        f2 = f2.squeeze()

        H, W, _ = image1.shape
        C, fH, fW = f1.shape
        assert fH == fW
        feat_dim = fH

        ratio = H / fH
        f1 = f1.reshape(C, fH * fW).T
        f2 = f2.reshape(C, fH * fW).T

        uv_c1 = torch.floor(uv_c1 / ratio).long().squeeze().numpy()
        uv_c2 = uv_c2.long().squeeze().numpy()

        dist = torch.mm(f1, f2.T)

        max_idx = torch.argmax(dist, dim=1)

        row1, col1 = uv_c1
        row2_gt, col2_gt = uv_c2

        i = int(row1 * fH + col1)
        row2, col2 = np.divmod(max_idx[i].item(), feat_dim)
        euc_dist = dist[i, :].reshape(fH, fW).cpu().numpy()
        distance_im = np.clip(euc_dist, 0.0, 1)

        row1 = np.floor(row1 * ratio).astype(int)
        col1 = np.floor(col1 * ratio).astype(int)

        row2 = np.floor(row2 * ratio).astype(int)
        col2 = np.floor(col2 * ratio).astype(int)

        img = np.zeros((H, W * 2, 3))
        im1 = (image1 * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        im2 = (image2 * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        im_tgt = (image2 * 255).astype(np.uint8)[:, :, [2, 1, 0]]

        im = copy.deepcopy(im1)
        img_out_1 = np.zeros(im.shape)
        img_out_1[:, :, :] = im[:, :, :]
        cv2.circle(img_out_1, (col1+int(ratio//2), row1+int(ratio//2)), 1, (0, 255, 255), thickness=2)
        cv2.rectangle(img_out_1, (col1, row1), (col1+int(ratio), row1+int(ratio)), (0, 255, 255), thickness=1)
         
        #leftmost image
        im = copy.deepcopy(im2)
        img_out_2 = np.zeros(im.shape)
        img_out_2[:, :, :] = im[:, :, :]
 
        #middle image
        im2 = (distance_im * 255).astype(np.uint8)
        im2 = cv2.resize(im2, (H, W), interpolation=cv2.INTER_NEAREST)
        im2 = np.stack([im2, im2, im2], axis=2)
        heatmap_img = cv2.applyColorMap(im2, cv2.COLORMAP_INFERNO)
        img_out_3 = cv2.addWeighted(heatmap_img, 0.7, im_tgt, 0.3, 0)
        cv2.circle(img_out_3, (col2+int(ratio/2), row2+int(ratio/2)), 1, (255, 0, 0), thickness=2)

        cv2.rectangle(img_out_3, (col2, row2), (col2+int(ratio), row2+int(ratio)), (255, 0, 0), thickness=1)

        # writing
        img[:, :W, :] = img_out_1
        img[:, W : 2 * W, :] = img_out_3

        return img

    def _get_map(image1, image2, uv_c1, uv_c2):

        uv_c1 = uv_c1[:, [0], :]
        uv_c2 = uv_c2[:, [0], :]

        feats1 = encoder.module(image1)
        feats2 = encoder.module(image2)

        local_feat_grid_1 = feats1['local_feat_pre_proj']
        local_feat_grid_2 = feats2['local_feat_pre_proj']

        mask_1 = _create_mask(image1, size = 32)
        mask_2 = _create_mask(image2, size = 32)

        local_feat_grid_1 = projector(local_feat_grid_1)
        local_feat_grid_2 = projector(local_feat_grid_2)

        local_feat_grid_1 = mask_1.unsqueeze(1) * local_feat_grid_1
        local_feat_grid_2 = mask_2.unsqueeze(1) * local_feat_grid_2

        mask_viz = torch.cat([torch.nn.functional.interpolate(mask_1.unsqueeze(1), (256,256)), \
                torch.nn.functional.interpolate(mask_2.unsqueeze(1), (256,256))], -1)

        outputs = []

        for (image1_i, 
            image2_i, 
            uv_c1_i, 
            uv_c2_i, 
            local_feat_grid_1_i, 
            local_feat_grid_2_i) \
            in zip(image1, 
                image2, 
                uv_c1, 
                uv_c2, 
                local_feat_grid_1, 
                local_feat_grid_2):
            
            img1 = (image1_i.cpu().squeeze().numpy().transpose(1, 2, 0)+1)/2
            img2 = (image2_i.cpu().squeeze().numpy().transpose(1, 2, 0)+1)/2

            outputs.append(viz_local_heatmaps(img1, img2, uv_c1_i, uv_c2_i, \
                local_feat_grid_1_i.detach(), local_feat_grid_2_i.detach()))

        return outputs, mask_viz

    img_v1, img_v2, pxy_v1, pxy_v2, img_c, pxy_c, imgs, mesh = batch['img_v1'], \
                                batch['img_v2'], batch['pxy_v1'], batch['pxy_v2'], \
                                batch['img_c'], batch['pxy_c'], batch['imgs'], batch['mesh'] 


    image1 = img_c.cuda()[:2]
    image2 = img_v2.cuda()[:2]
    uv_c1 = pxy_c[:2]
    uv_c2 = pxy_v2[:2]

    outputs2, mask_viz2 = _get_map(image1, image2, uv_c1, uv_c2)

    outs = []

    for pxy_c_i in pxy_c:

        image1 = img_c.cuda()[:1].repeat(mesh[0].shape[0],1,1,1)
        image2 = mesh[0].cuda()
        uv_c1 = pxy_c_i.unsqueeze(0).repeat(mesh[0].shape[0],1,1)
        uv_c2 = pxy_v2[:mesh[0].shape[0]]

        outputs3, mask_viz3 = _get_map(image1, image2, uv_c1, uv_c2)

        out = np.concatenate([outputs3[0][:,:256,:]] + [i[:,256:,:] for i in outputs3], -2)
        outs.append(out)

    return np.concatenate(outs,0)



def attention_map(batch, activation):

    colormap = plt.get_cmap('viridis') 

    saq, sak, caq, cak = activation['sa-q'], activation['sa-k'], activation['ca-q'], activation['ca-k']
    sa_attn = torch.softmax(torch.einsum('bic,bjc->bij', saq, sak)/256,-1)
    sa_attn = F.interpolate(sa_attn[:,0,1:].view(-1, 1, 32, 32), size=(256,256), mode = 'bilinear')
    sa_attn = (sa_attn - sa_attn.min()) / (sa_attn.max() - sa_attn.min())
    sa_attn = (sa_attn.cpu().numpy() * 255).astype('uint8')
    sa_attn = colormap(sa_attn[:,0])[:,:,:,:3]
    sa_attn = np.concatenate([sa_attn_i for sa_attn_i in sa_attn],0)[:,:,::-1]*255

    ca_attn = torch.softmax(torch.einsum('bic,bjc->bij', caq, cak)/math.sqrt(cak.shape[1]), -1)
    ca_attn = F.interpolate(rearrange(ca_attn[:,0], 'b (n hr wr)-> (b n) 1 hr wr', hr = 32, wr = 32,), size=(256,256), mode = 'bilinear')
    ca_attn =  rearrange( ca_attn[:,0], '(b n) hr wr-> b n hr wr', b = caq.shape[0])
    ca_attn = (ca_attn - ca_attn.min()) / (ca_attn.max() - ca_attn.min())
    ca_attn = (ca_attn.cpu().numpy() * 255).astype('uint8')
    ca_attn = np.concatenate([colormap(ca_attn[:,i])[:,:,:,:3] for i in range(ca_attn.shape[1])], 0)

    return sa_attn, ca_attn


def batch_prediction_visualize(batch, output_dict, add_pred_bbox = False):

    imgs = ((batch['imgs'].permute(0,2,3,1).detach().cpu().numpy()+1)/2)*255
    labels = output_dict['gt_label'].detach().cpu().numpy()  
    pred_labels = output_dict['pred_label'].detach().cpu().numpy()  
    cnfdnce = output_dict['pred'].sigmoid().detach().cpu().numpy()  
    gt_bbox = xywh2xyxy(batch['bbox']).detach().cpu().numpy()*256
    pred_bbox = xywh2xyxy(output_dict['pred_bbox'].sigmoid()).detach().cpu().numpy()*256
    anomaly_type = batch['anomaly_type']
    type_dict = {'position':1, 'rotate':2, 'missing':3, 'damaged':4, 'swapped':5}
    type_dict_inv = {j:i for i,j in type_dict.items()}

    outs = []
    outs_original = []
    for im_,lab_,pr_lab_,cnf_, bbox_, pr_bbox_, anm_type_ in zip(imgs, labels, pred_labels, cnfdnce, gt_bbox, pred_bbox, anomaly_type):
        perc_str = f'{round((cnf_[0])*100,2)}%' if pr_lab_== 1 else f'{round((1-cnf_[0])*100,2)}%'
        label_dict = {1:'Anom', 0:'Norm'}
        im_ = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
        outs_original.append(im_.copy())
        if add_pred_bbox:
          bb.add(im_, 0, 0, 255, 255,\
                  f' [gt:{label_dict[int(lab_)]}] [pred:{label_dict[int(pr_lab_)]}] [{perc_str}]',\
                  'green' if pr_lab_==lab_ else 'red')
        if int(lab_) == 1 and add_pred_bbox:
     
            bb.add(im_, *bbox_.astype('int'), type_dict_inv[anm_type_.item()], 'blue')

        if int(pr_lab_) == 1 and add_pred_bbox:
            bb.add(im_, *pr_bbox_.astype('int'), 'pr', 'maroon')
        outs.append(im_)
    outs = np.concatenate(outs, 0)
    outs_original = np.concatenate(outs_original, 0)

    return outs#np.concatenate([outs, outs_original], 0)

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def add_grid_line(image, color):
  image_height, image_width = image.shape[:2]
  for i in range(0, image_height, 8):
      cv2.line(image, (0, i), (image_width, i), color, 1, 1)
  for j in range(0, image_width, 8):
      cv2.line(image, (j, 0), (j, image_height), color, 1, 1)
  return image

def visualize_topk(batch, output_dict):
  rgb_colors = [[255,215,0], [255,102,102], [255,102,102]]
  anomaly_type = batch['anomaly_type']
  type_dict = {'position':1, 'rotate':2, 'missing':3, 'damaged':4, 'swapped':5}
  type_dict_inv = {j:i for i,j in type_dict.items()}

  mask = output_dict['topk_mask']
  affinity = output_dict['topk_dot']
  bbox = xywh2xyxy((batch['bbox']))*256
  bs, fhw, nfhw = mask.shape
  fh = fw = int(math.sqrt(fhw))
  H, W = batch['imgs'].shape[2:]
  ratio = H/fh

  px_points_all = []
  px_mask_all = []
  px_dist_all = []

  for _ in range(1):
    pixel_pnts = []
    q_pnts = []
    for bbox_i, lab_i, aff_ in zip(bbox, batch['labels'], affinity):
      if lab_i.item()==1:
        query_point = ((bbox_i[:2]+bbox_i[2:])/2)
        query_point = torch.Tensor([query_point[1], query_point[0]])
        pixel_pnts.append(query_point.clone())
        query_point = torch.floor(query_point / ratio).long().squeeze().numpy()
        col, row = query_point
        q_pnts.append(int(col * fh + row))
      else:
        fpix_i = np.random.choice(torch.argsort(aff_.sum(1), descending = True)[:32].cpu().numpy()) #np.random.choice(torch.where(aff_.sum(1)>0)[0].cpu().numpy())
        print (fpix_i)
        q_pnts.append(fpix_i)
        col, row = fpix_i//fh, fpix_i%fh
        query_point = torch.Tensor([col*ratio, row*ratio])
        pixel_pnts.append(query_point.clone())


    q_pnts = torch.Tensor(q_pnts)
    mask_q = (mask==0).float()[torch.arange(bs),q_pnts.long()]
    affinity_q = affinity.float()[torch.arange(bs),q_pnts.long()]
    dist_q = torch.clip(affinity_q, 0.0, 1)
    mask_q = mask_q*affinity_q
    mask_q = rearrange(mask_q, 'b (n h w) -> (b n) 1 h w', h = fh, w = fw)
    dist_q = rearrange(dist_q, 'b (n h w) -> (b n) 1 h w', h = fh, w = fw)
    mask_q = F.interpolate(mask_q, (256,256))
    dist_q = F.interpolate(dist_q, (256,256))
    mask_q = rearrange(mask_q, '(b n) 1 h w -> b 1 h (n w)', b = bs)
    dist_q = rearrange(dist_q, '(b n) 1 h w -> b 1 h (n w)', b = bs)
    px_points_all.append(pixel_pnts)
    px_mask_all.append(mask_q)
    px_dist_all.append(dist_q)


  mesh_imgs = rearrange(batch['mesh'], 'b n c h w -> b c h (n w)', b = bs)
  maps = []
  for i in range(len(batch['imgs'])):
    vis_maps = []
    imi = cv2.cvtColor((((batch['imgs'][i]+1)/2)*255).permute(1,2,0).cpu().numpy().astype('uint8'), cv2.COLOR_RGB2BGR)

    for mask_idx, (mask_q, pixel_pnts, dist_q) in enumerate(zip(px_mask_all, px_points_all, px_dist_all)):
      meshi = (((mesh_imgs[i]+1)/2)*255).permute(1,2,0).cpu().numpy().astype('uint8')
      maski = (mask_q[i][0].cpu().numpy()*255).astype('uint8')
      disti = (dist_q[i][0].cpu().numpy()*255).astype('uint8')
      #masked_imi = overlay(meshi, cv2.applyColorMap(maski, cv2.COLORMAP_INFERNO), color=[252, 102, 3], alpha=0.2, resize=None)
      dist_map = cv2.addWeighted(cv2.applyColorMap(disti, cv2.COLORMAP_INFERNO), 0.7, meshi, 0.3, 0)
      meshi = overlay(meshi, np.uint8(maski!=0)*255, rgb_colors[mask_idx], 0.9)#cv2.addWeighted(cv2.applyColorMap(maski, cv2.COLORMAP_INFERNO), 0.7, meshi, 0.3, 0)
      x_,y_ = pixel_pnts[i].int()
      imi[x_:x_+8, y_:y_+8]=rgb_colors[mask_idx][::-1]
      # try:
      #   bb.add(imi, *bbox[i].int(), type_dict_inv[anomaly_type[i].item()], color = 'red')
      # except:
      #   pass
      vis_map = np.concatenate([add_grid_line(imi, (240,240,240)), add_grid_line(meshi,(240,240,240)), dist_map],1)
      vis_maps.append(vis_map)
    maps.append(np.concatenate(vis_maps,0))
  return np.concatenate(maps,0)


def create_pose_parallel(batch, output_dict):
  bs = batch['imgs'].shape[0]
  from train import _create_mask
  mpf = rearrange(output_dict['mesh_proj_feats'], 'b (n h w) c -> b n (h w) c', h = 32, w = 32)
  qpf = output_dict['query_proj_feats']
  flat_ind = torch.einsum('bnic,bjc->bnij', mpf, qpf).argmax(-1).view(bs, -1, 32, 32)
  xy_tgt = torch.stack([flat_ind//32, flat_ind%32], -1)
  mask = _create_mask(batch['imgs'].cuda(), size = 32)
  flat_src = (torch.arange(1024).repeat(bs,1).view(bs, 32, 32).cuda()*mask).unsqueeze(1)
  xy_src = torch.stack([flat_src//32, flat_src%32], -1)
  pred_ind_pose = ((xy_src-xy_tgt)**2).sum(-1).sqrt().sum([2,3]).argmin(1)
  #pred_ind_pose = torch.stack([torch.cdist(qpf, mpf[:,i]).sum([1,2]) for i in range(mpf.shape[1])],1).argmin(1)
  best_matching_views = torch.stack([batch['mesh'][idx][i] for idx, i in enumerate(pred_ind_pose)], 0)
  vis_pose = torch.concat([batch['imgs'],best_matching_views], -1)
  return vis_pose
  #cv2.imwrite('im.png',(((vis_pose.permute(0,2,3,1).cpu().numpy()+1)/2)*255)[0])

def est_pose_parallel(batch, output_dict):

  bs = batch['imgs'].shape[0]
  from train import _create_mask
  mpf = rearrange(output_dict['mesh_proj_feats'], 'b (n h w) c -> b n (h w) c', h = 32, w = 32)
  qpf = output_dict['query_proj_feats']
  flat_ind = torch.einsum('bnic,bjc->bnij', mpf, qpf).argmax(-1).view(bs, -1, 32, 32)
  xy_tgt = torch.stack([flat_ind//32, flat_ind%32], -1)
  mask = _create_mask(batch['imgs'].cuda(), size = 32)
  flat_src = (torch.arange(1024).repeat(bs,1).view(bs, 32, 32).cuda()*mask).unsqueeze(1)
  xy_src = torch.stack([flat_src//32, flat_src%32], -1)
  pred_ind_pose = ((xy_src-xy_tgt)**2).sum(-1).sqrt().sum([2,3]).argmin(1)
  #pred_ind_pose = torch.stack([torch.cdist(qpf, mpf[:,i]).sum([1,2]) for i in range(mpf.shape[1])],1).argmin(1)
  q_poses = [i.split('_')[-3] for i in batch['path']]
  mesh_poses = [[j.split('_')[-2] for j in i][0] for i in batch['mesh_path']]
  gt_pose = torch.Tensor([20-np.array([abs(int(j)-int(i)) for j in mesh_poses]).argmin() for i in q_poses]).cuda() 
  gt_pose[gt_pose==20]=0
  top1_output_all = [i.item() for i in gt_pose == pred_ind_pose]
  top2_output_all = [abs(i-j)<2 for i,j in zip(gt_pose, pred_ind_pose)]
  top3_output_all = [abs(i-j)<3 for i,j in zip(gt_pose, pred_ind_pose)]
  top5_output_all = [abs(i-j)<5 for i,j in zip(gt_pose, pred_ind_pose)]
  #top1_output_normal = [i.item() for i,lab in zip(gt_pose == pred_ind_pose,batch['labels']) if lab == 1]

  return top1_output_all, top2_output_all, top3_output_all, top5_output_all

