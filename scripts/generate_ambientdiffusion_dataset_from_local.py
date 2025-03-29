import os
import torch
from tqdm import tqdm
from deepinv.datasets.fastmri import LocalFastMRISliceDataset
from deepinv.physics.mri import MRIMixin

if __name__ == "__main__":
    test_dataset  = LocalFastMRISliceDataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_8_16_test")
    data_dir = "/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_8_16_test_ambientdiffusion"

    os.makedirs(data_dir, exist_ok=True)

    ttc = lambda x: MRIMixin().to_torch_complex(x.unsqueeze(0)).squeeze(0) #2,(N,)H,W -> (N,)H,W

    for i in tqdm(range(len(test_dataset))):
        x, y, params = test_dataset[i]
        torch.save(
            {
                "gt": ttc(x),
                "ksp": ttc(y),
                "s_map": params["coil_maps"],
                "mask": params["mask"],
            },
            f"{str(data_dir)}/sample_{i}.pt",
        )

# python generate_ambientdiffusion_dataset_from_local.py