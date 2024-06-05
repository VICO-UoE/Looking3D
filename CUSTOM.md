## Testing using custom data 

For testing, you need two files as arguments ```--obj_path``` and ```--query_path```. ```obj_path``` can be a 3D mesh file in obj/stl/glb format, and ```query_path``` is the path of query image. 

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

You may need to normalize the mesh vertices within [-1,1] if its not already.

```utils/render_multiview.py``` renders multiview images (in .png format), and stores intrinsic and extrinsic camera matrices (in .json format) as well as correspondence matrices (in .npy format). During inference, the correspondence matrices are not used.


## Training/finetuning on custom data 

To finetune, you need to prepare your training data in a particular way to reuse our dataloader code. 

```
custom_data/
|
└──normals/
   └── <normal_1>/                            
       ├── rendered_mesh/  ---> [This is the output of `utils/render_multiview.py` given 3d mesh as input; consists of list of (.png, .json, .npy) files.]  
       ├── renders/      
       │   ├── <query_1.png>  

   anomalies/
   └── <anomaly_1>/                             
       ├── annotation/                                
       │   ├── <anno_1.json>  ---> [This file has bounding box details of anomolous region. You need this if you want to do localization. See our dataset for structure.]        
       ├── renders/                                  
           ├── <query_1.png>   
           └── ...

```


Otherwise, you can also use your own folder structure and rewrite the dataloader. In either case, the general step is to first render multiview images and associated files using ```utils/render_multiview.py```. You do not need to worry about camera intrinsics or the camera coordinate system, as these are handled by this file. The ```pos_enc3d``` is also generated (this is basically the .npy files).

Our model does not use the camera matrices (including intrinsic matrices) of the query images. So you only need the .png file and its bounding box annotation (the latter is optional, if localization is not performed).











