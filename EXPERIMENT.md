## Our Method - CMT: Correspondence Matching Transformer

Our model `Correspondence Matching Transformer` (CMT) learns to capture cross-modality relationships by applying a novel cross-attention mechanism through a sparse set of local correspondences. We use an auxiliary task that forces the model to learn viewpoint invariant representations for each local patch in query and multi-view images enabling our method to align local
features corresponding to the same 3D location regardless of its viewpoint without ground truth correspondences.

<img src=figures/diagram.jpg>

## Training

This code supports multi-GPU training. You can change `--nproc_per_node` to 1 for one GPU. To train the model from scratch:

```bash {"id":"01J228RV1PMHX9SVS8YWR0N95B"}
python -m torch.distributed.launch --nproc_per_node=4 --master_port 48949 train.py --exp_name CMT-final \
--data_path <BROKENCHAIRS_DATASET_PATH> --epochs 100 --lr 2e-5 --batch_size 6 --num_workers 4 --pred_box
```

To finetune the model run the following command:

```bash {"id":"01J228RV1PMHX9SVS8YX5K27N6"}
python -m torch.distributed.launch --nproc_per_node=4 --master_port 48949 train.py --exp_name CMT-finetune \
--data_path <BROKENCHAIRS_DATASET_PATH> --epochs 100 --lr 2e-5  --batch_size 6 \
--num_workers 4 --pred_box --resume_ckpt experiments/CMT-final/checkpoints/model.pt
```

## Testing on BrokenChairs-180K test set

```bash {"id":"01J228RV1PMHX9SVS8Z00EP3GV"}
python -m torch.distributed.launch --nproc_per_node=1 --master_port 42949 evaluate.py \
--data_path <BROKENCHAIRS_DATASET_PATH> --resume_ckpt experiments/CMT-final/checkpoints/model.pt \
--num_mesh_images 20  --batch_size 6 --num_workers 4 --pred_box
```
