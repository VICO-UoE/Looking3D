import os
import cv2 
import math, json, glob
import numpy as np
from io import BytesIO
from PIL import Image
from itertools import cycle
import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py
import albumentations as A

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return sample_inds

def read_json_KRT(fname):
    with open(fname,'r') as f:
        data = json.load(f)

    KRT = torch.Tensor((np.array(data['K'])@np.array(data['RT'][:3])))
    return KRT

def get_transform(size=256, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    
    
    transform_list.append(transforms.Resize(size, interpolation=method))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class dataloader(Dataset):

    def __init__(self, data_path, num_mesh_images = 5, mode = 'train'):

        self.all_image_paths = glob.glob(f'{data_path}/images/*/*')
        self.shape_path = f'{data_path}/shapes/'
        self.splits =  json.load(open(f'{data_path}/split.json','r'))
        
        self.paths = [i for i in self.all_image_paths if os.path.basename(i).split('_')[1] in self.splits[mode]]
    
        self.mode = mode
        self.num_mesh_images = num_mesh_images
        self.n_pnts = 32

        size = 256
        self.transform_box = A.Compose([
            A.Resize(size, size),
            A.RandomCrop(width=224, height=224, p = 1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(size, size),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.transform_test = A.Compose([
            A.Resize(size, size),
            A.RandomCrop(width=224, height=224, p = 0.2),
            A.HorizontalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(size, size),
            A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # split_dict = {'train': [i['anno_id'] for i in json.load(open('data/splits/Chair.train.json','r'))],
        #             'test' : [i['anno_id'] for i in json.load(open('data/splits/Chair.test.json','r'))],
        #             'val' :  [i['anno_id'] for i in json.load(open('data/splits/Chair.val.json','r'))]}
        # self.type_dict = {'position':1, 'rotate':2, 'missing':3, 'damaged':4, 'swapped':5}

        # good_train_list, good_test_list, good_val_list = [], [], []
        # bad_train_list, bad_test_list, bad_val_list = [], [], []

        # for i in good_samples_list:
        #     if i.split('/')[-3] in split_dict['test']:
        #         good_test_list.append(i)
        #     elif i.split('/')[-3] in split_dict['val']:
        #         good_val_list.append(i)
        #     else:
        #         good_train_list.append(i)

        # for i in bad_samples_list:
        #     if i.split('/')[-3].split('_')[0][1:] in split_dict['test']:
        #         bad_test_list.append(i)
        #     elif i.split('/')[-3].split('_')[0][1:] in split_dict['val']:
        #         bad_val_list.append(i)
        #     else:
        #         bad_train_list.append(i)

        # if mode == 'train':
        #     self.good_list, self.bad_list = good_train_list, bad_train_list
        #     self.good_list_upsampled = self.good_list
        # elif mode == 'test':
        #     self.good_list, self.bad_list = good_test_list, bad_test_list
        #     self.good_list_upsampled = self.good_list
        # elif mode == 'val':
        #     self.good_list, self.bad_list = good_val_list, bad_val_list
        #     self.good_list_upsampled = self.good_list
        
        
        # self.paths = self.good_list_upsampled + self.bad_list
        # self.labels = [0]*len(self.good_list_upsampled) + [1]*len(self.bad_list)

        # if self.args.pretraining: self.objs = glob.glob('/disk/scratch_ssd/s2514643/brokenchairs/normals/*')

        # print (f'#dataset:[{mode}] total {len(self.good_list)+len(self.bad_list)} -> {len(self.good_list)} good and {len(self.bad_list)} bad images')


    def __len__(self):  
        
        return len(self.paths)

    def get_image_tensor(self, path):
        img = Image.open(path)
        trans = get_transform(size = 256)
        img = trans(img)
        return img    

    def get_mesh_image_tensor(self, path):
        img = Image.open(path).convert('RGB')
        trans = get_transform(size = 256)
        img = trans(img)
        return img    

    def to_image_tensor(self, input, aug = True):

        if not aug:
            if isinstance(input, list):
                img_path, bbox = input
            else:
                img_path = input
            imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            transformed = self.transform_test(image = imgs)
            imgs = transformed['image']
            imgs = transforms.ToTensor()(imgs)
            if isinstance(input, list):
                return imgs, bbox
            else:
                return imgs
                
        else:
            if isinstance(input, list):
                img_path, bbox = input
                imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                transformed = self.transform_box(image = imgs, bboxes = bbox.unsqueeze(0), class_labels = ['none'])
                imgs, bbox = transformed['image'], torch.Tensor(transformed['bboxes']).squeeze()
                imgs = transforms.ToTensor()(imgs)
                return [imgs, bbox]
            else:
                img_path = input
                imgs = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                transformed = self.transform_box(image = imgs,  bboxes = [], class_labels = [])
                imgs = transformed['image']
                imgs = transforms.ToTensor()(imgs)
                return imgs

    def __getitem__(self, index):

        try:
            img_path = self.paths[index]
            filename = os.path.basename(img_path)
            model_id = filename.split('_')[1]

            ref_mesh_path = glob.glob(self.shape_path + f'/shape_{model_id}/mv_images/*.png')
            ref_mesh_path_select = np.random.choice(ref_mesh_path, self.num_mesh_images)
            mesh_images = torch.stack([self.get_mesh_image_tensor(i) for i in ref_mesh_path_select if i.endswith('.png')],0)
            mesh_pos_enc3d = torch.stack([torch.Tensor(np.load(i.replace('.png', '.npy'), allow_pickle = True).item()['xyz']) for i in ref_mesh_path_select])


            if 'anomaly' in filename:
                bad_json_path = img_path.replace('/images/', '/annotations/').replace('/render_','/info_').replace('.png','.json')
                with open(bad_json_path, 'r') as f:
                    info = json.load(f)
                    xi,yi,hi,wi = np.array(info['2d_bbox'])/512
                    xi = xi+hi/2
                    yi = yi+wi/2
                    bbox = torch.Tensor([xi,yi,hi,wi])
                query_imgs, bbox = self.to_image_tensor([img_path, bbox], aug = True)
                labels = 1
                if bbox.shape[0] != 4: pass #print ('WRN: /data/error - ignoring...')
                assert bbox.shape[0] == 4
            else:
                bbox = torch.Tensor([0,0,0,0])
                query_imgs = self.to_image_tensor(img_path, aug = True)
                labels = 0


            #### below codes only requires for training the VLFA block
            c=0
            while True:
                random_2_views = np.random.choice(ref_mesh_path, 2, replace = False)
                color_img_path = img_path

                xy_data_v1, xy_data_v2 = [np.load(i.replace('.png','.npy'), allow_pickle=True).item() for i in random_2_views]
                paired_idx = np.where(xy_data_v1['is_visible'] & xy_data_v2['is_visible'])[0]
                c=c+1
                if len(paired_idx)>=self.n_pnts:
                    break
                if c>10:
                    return None
            
            pxy_v1, pxy_v2 = xy_data_v1['pxy'][paired_idx], xy_data_v2['pxy'][paired_idx]
            n_far_indx = fps(pxy_v1, self.n_pnts)
            pxy_v1, pxy_v2 = torch.Tensor(pxy_v1[n_far_indx]), torch.Tensor(pxy_v2[n_far_indx])

            img_v1 = self.get_mesh_image_tensor(random_2_views[0]) 
            img_v2 = self.get_mesh_image_tensor(random_2_views[1]) 
            img_c = self.to_image_tensor(color_img_path, aug = True)

                
            return_dict = {'query_imgs':query_imgs, 
                           'mesh_images':mesh_images, 
                           'mesh_pos_enc3d': mesh_pos_enc3d,
                           'labels':labels, 
                           'bbox':bbox, 
                           'img_v1':img_v1, 
                           'img_v2':img_v2, 
                           'pxy_v1':pxy_v1, 
                           'pxy_v2':pxy_v2, 
                           'img_c':img_c, 
                           'path':img_path}

            return return_dict 

        except:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def data_sampler(dataset, shuffle, distributed):

    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)

def get_dataloaders(args, num_mesh_images = [5,5]):

    # if not os.path.isdir(args.data_path):
    #     args.data_path = '/raid/s2514643/brokenchairs/'
        

    train_dataset = dataloader(args.data_path, num_mesh_images = num_mesh_images[0], mode = 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers,
    )          

    test_dataset = dataloader(args.data_path, num_mesh_images = num_mesh_images[1], mode = 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(test_dataset, shuffle=True, distributed=args.distributed),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )      

    return train_loader, test_loader 



# train_dataset = dataloader(10, 'train')
# train_loader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size = 1,
#     collate_fn=collate_fn,
#     drop_last=True,
#     num_workers=1,
# )   

# data = next(iter(train_loader))