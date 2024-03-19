from train import forward_cmt, build_network
import numpy as np
import albumentations as A
from PIL import Image
import torchvision.transforms as transforms
import glob, uuid, os, cv2, torch
from utils.visualize import batch_prediction_visualize, attention_map, visualize_topk, create_pose_parallel,est_pose_parallel


def load_itw_samples(query_path, mv_path, device):

    """
    Parameters:
        query_path (str): Path to the query images directory OR the query image file.
        mv_path (str): Path to the multiview images directory (images should be in .pngs)
        device (str): Device to load the samples onto ('cuda' or 'cpu').
    """
    img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)

    def get_transform(size=256, method=Image.BICUBIC, normalize=True, toTensor=True):
        transform_list = []
        transform_list.append(transforms.Resize(size, interpolation=method))
        if toTensor:
            transform_list += [transforms.ToTensor()]
        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def get_image_tensor(path):
        img = Image.open(path).convert('RGB')
        trans = get_transform(size = 256)
        img = trans(img)
        return img   

    transform_box = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(width=224, height=224, p = 1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(256, 256),
        A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    transformed = transform_box(image = img,  bboxes = [], class_labels = [])
    mv_images = torch.stack([get_image_tensor(i) for i in glob.glob(mv_path+'/*') if i.endswith('.png')],0)
    pos_enc3d = torch.stack([torch.Tensor(np.load(i.replace('.png', '.npy'), allow_pickle = True).item()['xyz']) for i in glob.glob(mv_path+'/*.png')])
    batch = {'imgs':transforms.ToTensor()(transformed['image']).unsqueeze(0).to(device), \
        'mesh':mv_images.unsqueeze(0).to(device), 'labels':torch.Tensor(1).to(device), \
            'bbox':torch.Tensor([0,0,0,0]).unsqueeze(0).to(device), \
                'pos_enc3d':pos_enc3d.unsqueeze(0).to(device), 'anomaly_type':'position'}

    return batch

def predict(query_path, mv_path, resume_ckpt, device, topk = 100):

    """
    Parameters:
        query_path (str): Path to the query images directory OR the query image file.
        mv_path (str): Path to the multiview images directory. (images should be in .pngs)
        resume_ckpt (str): Path to the checkpoint file for model resuming.
        device (str): Device to perform inference ('cuda' or 'cpu').
        topk (int): Number of top matches to return (default is 100).

    """

    models = build_network(2, device, distributed = False, resume_ckpt = resume_ckpt, local_rank = 0)
    [model.eval() for model in models]

    if os.path.isdir(query_path):
        queries = [i for i in glob.glob(query_path+'/*') if (i.endswith('.jpg') or i.endswith('.png'))]
    else:
        queries = [query_path]

    pred_labels = []

    for qidx, query_i in enumerate(queries):
        
        batch = load_itw_samples(query_i, mv_path, device)

        result = forward_cmt(batch, models, is_train = False, topk = topk)
        pred_label = int(result['pred_label'].item())
        conf = 100 - 100*torch.sigmoid(result['pred']).item() if pred_label == 0 else 100*torch.sigmoid(result['pred']).item()

        pred_labels.append(pred_label)

        print (f'-> Query_path : {query_i}\n Anoamly_pred_label : {pred_label}\nConf_score : {conf}')

    return pred_labels
    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')

    parser.add_argument('--mv_path', type=str)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    predict(args.query_path, args.mv_path, args.resume_ckpt, args.device, args.topk)