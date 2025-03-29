import os
from argparse import ArgumentParser
import torch

parser = ArgumentParser()
parser.add_argument("--acc", type=int, default=8)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()

measurements_path = f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_16_test_ambientdiffusion"
outdir = f"/home/s2558406/RDS/models/deepinv-selfsup-fastmri/ambient-diffusion/results_{args.acc}"

for sample in range(len(os.listdir(measurements_path))):
    results_dir = outdir + "/trained_r=%d_delta_prob%d/sample%d/seed%d/R=%d"%(args.acc, args.acc + 1, sample, args.seed, args.acc)
    results = torch.load(results_dir + '/checkpoint.pt')
    print(
        'gt_img', results["gt_img"].shape,
        'recon', results["cplx_recon"].shape,
        'adj_img', results["adj_img"].shape,
        'nrmse', results["img_nrmse"],
        'ssim', results["img_SSIM"],
        'psnr', results["img_PSNR"]
    )

    #TODO