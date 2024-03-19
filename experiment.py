from train import *
import numpy as np
from tqdm import tqdm
from utils.visualize import batch_prediction_visualize, attention_map, visualize_topk, create_pose_parallel,est_pose_parallel
import argparse
import uuid
from PIL import Image

parser = argparse.ArgumentParser(description='help')
parser.add_argument('--n_pnts', type=int, default=32)
parser.add_argument('--topk', type=int, default=100)
parser.add_argument('--num_mesh_images', type=int, default=5)
parser.add_argument('--pred_box', action='store_true')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--distributed', type=bool, default=True)
parser.add_argument('--n_machine', type=int, default=1)
parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--resume_ckpt', type=str, default=None)
parser.add_argument('--data_path', type=str, default='/disk/scratch_ssd/s2514643/brokenchairs/')

args = parser.parse_args()
init_distributed()
local_rank = int(os.environ['LOCAL_RANK'])

args.resume_ckpt = 'experiments/final-augxcyc-longrun-bbox/checkpoints/model_000098.pt'

args.topk = 100
args.num_mesh_images = 20
args.batchsize = 6
_, test_dataset = get_dataloaders(args, num_mesh_images = [-1,args.num_mesh_images])
test_dataset = iter(test_dataset)
models = build_network(args, local_rank)

##add hooks to 
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook
[encoder, encoder_ema, projector, projector_ema, attention_decoder, local_contrast] = models
hooks = [attention_decoder.module.crossatten.transformer_blocks[-1].attn1.to_q.register_forward_hook(getActivation('sa-q')),
    attention_decoder.module.crossatten.transformer_blocks[-1].attn1.to_k.register_forward_hook(getActivation('sa-k')),
    attention_decoder.module.crossatten.transformer_blocks[-1].attn2.to_q.register_forward_hook(getActivation('ca-q')),
    attention_decoder.module.crossatten.transformer_blocks[-1].attn2.to_k.register_forward_hook(getActivation('ca-k')),
    projector.module.register_forward_hook(getActivation('proj_feats'))]

# ## visualize prediction
# out_dir = os.path.join(os.path.dirname(os.path.dirname(args.resume_ckpt)), 'results')
# os.makedirs(out_dir, exist_ok=True)
# for _ in tqdm(range(100)):
#     batch = next(test_dataset)
#     with torch.no_grad():
#         output_dict = forward_cmt(batch, models, is_train = False, topk = args.topk)
#     vis_img = batch_prediction_visualize(batch, output_dict, add_pred_bbox = False)
#     topk_vis_map = visualize_topk(batch, output_dict)
#     attn_map, ca_attn = attention_map(batch, activation)
#     im =  visualize_topk(batch, output_dict)
#     all_concatenated = np.concatenate([vis_img, attn_map, topk_vis_map], 1)
#     filename = os.path.join(out_dir, str(uuid.uuid4())+'.png')
#     cv2.imwrite(filename, all_concatenated)

# [_.remove() for _ in hooks]


# out_dir = os.path.join(os.path.dirname(os.path.dirname(args.resume_ckpt)), 'pose')
# os.makedirs(out_dir, exist_ok=True)
# for _ in tqdm(range(1000)):
#     batch = next(test_dataset)
#     with torch.no_grad():
#         output_dict = forward_cmt(batch, models, is_train = False, topk = args.topk)
#     vis_img = create_pose_parallel(batch, output_dict)
#     for batch_idx in range(batch['imgs'].shape[0]):
#       filename = os.path.join(out_dir, batch['path'][batch_idx].split('/')[-1])
#       cv2.imwrite(filename, (((vis_img.permute(0,2,3,1).cpu().numpy()+1)/2)*255)[batch_idx])
# [_.remove() for _ in hooks]

top1_acc_all, top2_acc_all, top3_acc_all, top5_acc_all = [], [], [], []

for batch in tqdm(test_dataset):
    with torch.no_grad():
        output_dict = forward_cmt(batch, models, is_train = False, topk = args.topk)
    top1_output_all, top2_output_all, top3_output_all, top5_output_all = est_pose_parallel(batch, output_dict)
    top1_acc_all.extend(top1_output_all)
    top2_acc_all.extend(top2_output_all)
    top3_acc_all.extend(top3_output_all)
    top5_acc_all.extend(top5_output_all)

    print (torch.Tensor(top1_acc_all).mean().item(), torch.Tensor(top2_acc_all).mean().item(), torch.Tensor(top3_acc_all).mean().item(), torch.Tensor(top5_acc_all).mean().item())

[_.remove() for _ in hooks]