# Looking 3D: Anomaly Detection with 2D-3D Alignment

<table>
  <tr>
      <strong><a href="https://arxiv.org/abs/xxx.xxxxx">Looking 3D: Anomaly Detection with 2D-3D Alignment</a></strong><br>
      Ankan Bhunia, Changjian Li, Hakan Bilen<br>
      CVPR 2024
  </tr>
</table>



[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://groups.inf.ed.ac.uk/vico/research/Looking3D)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://openaccess.thecvf.com/content/CVPR2024/papers/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.pdf)
[![dataset](https://img.shields.io/badge/Dataset-link-blue)](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing)


<hr />

## Dataset

<img src=figures/teaser.gif>

 - Download ```BrokenChairs180K.tar.gz``` from [here](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing).
 - The dataset contains around 180K rendered images with 100K classified as anomaly and 80K normal.
 - Please see ```DATA.md```  for folder structure.


## Our novel task & Conditional AD benchmark
Standard Anomaly Detection (AD) frameworks perform pooly without clear defination of ‘normality’, especially when abnormalities are arbitrary and instance-specific. Our paper introduces a novel conditional AD task, along with a new benchmark and an effective solution, that aims to identify and localize anomalies from a photo of an object instance (i.e., the query image), in relation to a reference 3D model. The 3D model provides the reference shape for the regular object instance, and hence a clear definition of regularity for the query image. This setting is motivated by real-world applications in inspection and quality control, where an object instance is manufactured based on a reference 3D model, which can then be used to identify anomalies (i.e., production faults, damages) from a photo of the instance.


<table>
  <tr>
    <td style="text-align: center;">
      <img src="figures/left.jpeg" alt="Image 1 Description" width="300" />
    </td>
    <td style="text-align: center;">
      <img src="figures/right.jpeg" alt="Image 2 Description" width="300" />
    </td>
  </tr>
</table>

## Method

Our model ```Correspondence Matching Transformer``` (CMT) learns to capture cross-modality relationships by applying a novel cross-attention mechanism through a sparse set of local correspondences. We use an auxiliary task that forces the model to learn viewpoint invariant representations for each local patch in query and multi-view images enabling our method to align local
features corresponding to the same 3D location regardless of its viewpoint without ground truth correspondences.

## Pretrained Models
 - Download the pretrained model from [here](https://drive.google.com/drive/folders/1D9YFDP0kJkojBa1Rb-fM2uAZoS_1Pm3G?usp=sharing) and place the .pt file inside ```experiments/CMT-final/checkpoints/```

   
## Conda Installation

```bash
# Create a conda virtual environment for basic training/testing: 
conda create -n Looking3D python=3.8
conda activate Looking3D
pip install opencv-python wandb tqdm albumentations einops h5py kornia bounding_box matplotlib omegaconf trimesh[all] xformers

# install pyrender and pytorch3d (optional; only required for rendering multiview images)
pip install pyrender
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Training and Evaluation on BrokenChairs-180K

Please see ```EXPERIMENT.md``` for commands to run the training and testing codes. 


## Inference

You can use the python function ```predict(.)``` in the ```demo.py``` file for the inference.

  ```python
from demo import predict

pred_labels = predict(query_path = "sample/query_example.png", \
                       mv_path = "sample/mv_images/", \
                       resume_ckpt = "experiments/CMT-final/checkpoints/model.pt", device = "cuda", topk = 100)
  ```

Please see ```CUSTOM.md``` for a detailed discussion on how to use the codebase to test and train on custom datasets containing shape-image pairs.

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
