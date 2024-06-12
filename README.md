# Looking 3D: Anomaly Detection with 2D-3D Alignment

<table>
  <tr>
      <strong><a href="https://arxiv.org/abs/xxx.xxxxx">Looking 3D: Anomaly Detection with 2D-3D Alignment</a></strong><br>
      Ankan Bhunia, Changjian Li, Hakan Bilen<br>
      CVPR 2024
  </tr>
</table>



[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://xxx.xxxxx)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/xxx.xxxxx)
[![dataset](https://img.shields.io/badge/Dataset-link-blue)](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing)


<hr />

## Dataset

<img src=figures/data.gif>

 - Download ```BrokenChairs180K.tar.gz``` from [here](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing).
 - The dataset contains around 180K rendered images with 100K classified as anomaly and 80K normal.
 - Please see ```DATA.md```  for folder structure.


## Our novel task: Conditional AD
   
<img src=figures/cad.jpeg>

## Pretrained Models
 - Download the pretrained model from [here](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing) and place the .pt file inside ```experiments/CMT-final/checkpoints/```

   
## Conda Installation

```bash
# Create a conda virtual environment for basic training/testing: 
conda create -n Looking3D python=3.8
conda activate Looking3D```DATA.md```
pip install opencv-python wandb tqdm albumentations einops h5py kornia bounding_box matplotlib omegaconf trimesh[all] xformers

# install pyrender and pytorch3d (optional; only required for rendering multiview images)
pip install pyrender
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```


## Method

<img src=figures/diagram.jpg>

## Training
This code supports multi-GPU training. You can change ```--nproc_per_node``` to 1 for one GPU. To train the model from scratch:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 48949 train.py --exp_name CMT-final \
--data_path <BROKENCHAIRS_DATASET_PATH> --epochs 100 --lr 2e-5 --batch_size 6 --num_workers 4 --pred_box
```
 To finetune the model run the following command:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 48949 train.py --exp_name CMT-finetune \
--data_path <BROKENCHAIRS_DATASET_PATH> --epochs 100 --lr 2e-5  --batch_size 6 \
--num_workers 4 --pred_box --resume_ckpt experiments/CMT-final/checkpoints/model.pt
```

## Testing on BrokenChairs-180K test set
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 42949 evaluate.py \
--data_path <BROKENCHAIRS_DATASET_PATH> --resume_ckpt experiments/CMT-final/checkpoints/model.pt \
--num_mesh_images 20  --batch_size 6 --num_workers 4 --pred_box
```

## Inference

You can use the python function ```predict(.)``` in the ```demo.py``` file for the inference.

  ```python
from demo import predict

pred_labels = predict(query_path = "sample/query_example.png", \
                       mv_path = "sample/mv_images/", \
                       resume_ckpt = "experiments/CMT-final/checkpoints/model.pt", device = "cuda", topk = 100)
  ```
To render the multiview images, you can use the following commands:


```bash
# Step 1. Render grayscale multiview images (pytorch3d and pyrender required)
python utils/render_multiview.py \
  --obj_path data/in-the-wild-examples/mesh_reference.glb \
  --out_path data/itw_testing/sample/mv_images/ \
  --num_imgs 20 \
  --gray_scale

# Step 2. Test the model
python demo.py --mv_path data/itw_testing/sample/mv_images/ \
  --query_path data/in-the-wild-examples/query_example.png \
  --resume_ckpt experiments/CMT-final/checkpoints/model.pt
```


## Citation

If you use the results and code for your research, please cite our paper:

```
@article{bhunia2024look3d,
  title={Looking 3D: Anomaly Detection with 2D-3D Alignment},
  author={Bhunia, Ankan Kumar and Li, Changjian and Bilen, Hakan},
  journal={CVPR},
  year={2024}
}
```
