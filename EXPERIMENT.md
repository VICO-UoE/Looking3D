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
