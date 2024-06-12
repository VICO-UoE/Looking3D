
```
BrokenChairs180K/
|
└──normals/
   └── 174/                                   # partnet folder id
       ├── rendered_mesh/                     # rendered mv images (Grayscale). 
       │   ├── 174_3.0_0_20.png               # filename of mv image.
       │   ├── 174_3.0_0_20.json              # Json file containing intristic and extrinsic parameters of the rendered image.
       │   ├── 174_3.0_0_20.npy               # npy file containing 2D-3D correspondence points. More details below.
       │   └── ...
       ├── renders/                           # rendered query images. 
       │   ├── 174_5730_2.5_180_35_0.png      # filename of the query (normal) image (RGB)
       │   └── ...
       ├── model.obj                          # .obj file of the model (textureless)

   anomalies/
   └── A1293_position/                                   # Anomaly folder name. format A{partnet_id}_{anomaly-type}
       ├── annotation/                                   # Contains all relevent annotations
       │   ├── info_1293_6034_2.5_0_30_5.json            # <info_>: It contains 2d_bbox, IoU, camera_parameters and examplar_id. 
       │   ├── mask_new_1293_6034_2.5_0_30_5.png         # <mask_new_>: binary mask of the object part after adding an anomaly. 
       │   ├── mask_old_1293_6034_2.5_0_30_5.png         # <mask_old_>: binary mask of the object part before adding an anomaly. 
       │   ├── vis_1293_6034_2.5_0_30_5.png              # some visualisation with boundary box and masking. 
       │   └── ...
       ├── renders/                                      # contains all rendered images. 
           ├── 1293_6034_2.5_0_30_5.png                  # filename of the query (anomaly) image (RGB)
           └── ...

```
