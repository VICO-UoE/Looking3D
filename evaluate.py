from train import *
import numpy as np


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='help')

    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='/disk/scratch_ssd/s2514643/brokenchairs/')
    parser.add_argument('--num_mesh_images', type=int, default=5)
    parser.add_argument('--n_pnts', type=int, default=32)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--pred_box', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=0)
    

    args = parser.parse_args()
    init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])

    _, test_dataset = get_dataloaders(args, num_mesh_images = [-1,args.num_mesh_images])
    models = build_network(args, local_rank)
    [model.eval() for model in models]
    result = test(args, test_dataset, models)
    res_tnsr_values = torch.Tensor([i for i in result.values()]).cuda()
    gathered_samples = [torch.zeros_like(res_tnsr_values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, res_tnsr_values) 
    if is_main_process():
        res = sum(gathered_samples)/len(gathered_samples)
        final_result = {i:round(res[idx].cpu().item(),4) for idx,i in enumerate(result.keys())}
        print ('topk: '+str(args.topk))
        print (final_result)
