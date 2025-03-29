import os
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from deepinv.datasets.fastmri import LocalDataset
from deepinv.physics.mri import MRIMixin

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--n_coils", type=int, default=16)
    args = parser.parse_args()

    test_dataset  = LocalDataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_test")
    data_dir = f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_test_ambientdiffusion"

    os.makedirs(data_dir, exist_ok=True)

    ttc = lambda x: MRIMixin().to_torch_complex(x.unsqueeze(0)).squeeze(0) #2,(N,)H,W -> (N,)H,W

    for i in tqdm(range(len(test_dataset))):
        x, y, params = test_dataset[i]
        torch.save(
            {
                "gt": ttc(x),
                "ksp": ttc(y),
                "s_map": params["coil_maps"],
                f"mask_{args.acc}": params["mask"],
            },
            f"{str(data_dir)}/sample_{i}.pt",
        )

# python scripts/ambient_diffusion_mri/generate_ambientdiffusion_dataset_from_local.py --acc 8