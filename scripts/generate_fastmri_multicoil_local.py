from pathlib import Path
from argparse import ArgumentParser
import torch

import deepinv as dinv
from deepinv.datasets import FastMRISliceDataset
from deepinv.datasets.fastmri import FullMultiCoilFastMRITransform
from deepinv.physics.generator import RandomMaskGenerator

def get_dataset(fnames, mask_generator, img_size, n_coils, slice_index="middle"):
    return FastMRISliceDataset(
        "/home/s2558406/RDS/data/fastmri/brain/multicoil_train",
        slice_index=slice_index,
        transform=FullMultiCoilFastMRITransform(
            mask_generator=mask_generator,
            crop_size=img_size
        ),
        filter_id=lambda s: s.metadata["coils"] == n_coils and Path(s.fname).stem in fnames,
        load_metadata_from_cache=True,
        save_metadata_to_cache=True,
        metadata_cache_file="/home/s2558406/RDS/data/fastmri/brain/multicoil_train_cache_new.pkl"
    )

if __name__ == "__main__":
    # TODO insert here fname extraction from cache from fastmri-experiments

    parser = ArgumentParser()
    parser.add_argument("--acc", type=int, default=8)
    parser.add_argument("--n_coils", type=int, default=16)
    parser.add_argument("--img_size", type=tuple, default=(384, 320))
    parser.add_argument("--center_fraction", type=float, default=0.0625) # = acs / W
    parser.add_argument("--rss", action="store_true")
    args = parser.parse_args()

    train_fnames = ['file_brain_AXT2_201_2010602', 'file_brain_AXT2_210_2100260', 'file_brain_AXT2_204_2040009', 'file_brain_AXT2_200_2000175', 'file_brain_AXT2_200_2000362', 'file_brain_AXT2_210_6001617', 'file_brain_AXT2_205_2050080', 'file_brain_AXT2_200_2000360', 'file_brain_AXT2_201_2010029', 'file_brain_AXT2_202_2020022', 'file_brain_AXT2_200_6002037', 'file_brain_AXT2_200_2000621', 'file_brain_AXT2_200_6002100', 'file_brain_AXT2_201_2010168', 'file_brain_AXT2_205_6000091', 'file_brain_AXT2_202_2020016', 'file_brain_AXT2_210_2100064', 'file_brain_AXT2_210_6001518', 'file_brain_AXT2_210_6001863', 'file_brain_AXT2_200_6002262', 'file_brain_AXT2_200_6002228', 'file_brain_AXT2_200_2000417', 'file_brain_AXT2_200_2000566', 'file_brain_AXT2_200_2000020', 'file_brain_AXT2_200_6002116', 'file_brain_AXT2_210_2100135', 'file_brain_AXT2_201_2010532', 'file_brain_AXT2_202_2020067', 'file_brain_AXT2_200_6002381', 'file_brain_AXT2_210_2100201', 'file_brain_AXT2_200_2000485', 'file_brain_AXT2_204_2040042', 'file_brain_AXT2_204_2040073', 'file_brain_AXT2_202_2020065', 'file_brain_AXT2_201_2010156', 'file_brain_AXT2_205_6000022', 'file_brain_AXT2_210_6001709', 'file_brain_AXT2_204_2040011', 'file_brain_AXT2_200_6002446', 'file_brain_AXT2_210_6001622', 'file_brain_AXT2_202_2020261', 'file_brain_AXT2_209_6001384', 'file_brain_AXT2_201_2010174', 'file_brain_AXT2_200_6002375', 'file_brain_AXT2_210_6001763', 'file_brain_AXT2_200_6002601', 'file_brain_AXT2_210_2100343', 'file_brain_AXT2_200_2000290', 'file_brain_AXT2_210_6001747', 'file_brain_AXT2_210_6001651', 'file_brain_AXT2_200_2000600', 'file_brain_AXT2_202_2020063', 'file_brain_AXT2_209_2090215', 'file_brain_AXT2_200_6002302', 'file_brain_AXT2_205_2050062', 'file_brain_AXT2_210_2100306', 'file_brain_AXT2_210_2100345', 'file_brain_AXT2_202_2020183', 'file_brain_AXT2_206_2060079', 'file_brain_AXT2_200_6002257', 'file_brain_AXT2_200_2000486', 'file_brain_AXT2_200_6002387', 'file_brain_AXT2_210_2100334', 'file_brain_AXT2_210_6001677', 'file_brain_AXT2_200_6002111', 'file_brain_AXT2_210_6001737']
    test_fnames = ['file_brain_AXT2_201_2010604', 'file_brain_AXT2_205_2050130', 'file_brain_AXT2_200_2000092', 'file_brain_AXT2_210_6001534', 'file_brain_AXT2_200_6002524', 'file_brain_AXT2_202_2020039', 'file_brain_AXT2_201_2010201', 'file_brain_AXT2_202_2020189', 'file_brain_AXT2_200_2000332', 'file_brain_AXT2_209_2090358', 'file_brain_AXT2_202_2020409', 'file_brain_AXT2_200_6002528', 'file_brain_AXT2_200_6002655', 'file_brain_AXT2_200_2000080', 'file_brain_AXT2_206_2060058', 'file_brain_AXT2_210_6001524', 'file_brain_AXT2_201_2010455']

    rng_cpu = torch.Generator().manual_seed(1)

    mask_generator = RandomMaskGenerator(
        img_size=args.img_size,
        acceleration=args.acc,
        center_fraction=args.center_fraction,
        rng=rng_cpu
    )
    assert mask_generator.n_center == 20

    train_dataset = get_dataset(train_fnames, mask_generator, args.img_size, args.n_coils, slice_index="middle+1")
    print(len(train_dataset)) #66*3
    train_dataset.save_local_dataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_train")

    test_dataset  = get_dataset(test_fnames,  mask_generator, args.img_size, args.n_coils, slice_index="middle")
    print(len(test_dataset)) #83-66=17
    test_dataset.save_local_dataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_test")

# python scripts/generate_fastmri_multicoil_local.py --acc 8
# python scripts/generate_fastmri_multicoil_local.py --acc 4 --rss