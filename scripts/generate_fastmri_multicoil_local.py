from pathlib import Path
from argparse import ArgumentParser
import torch

import deepinv as dinv
from deepinv.datasets import FastMRISliceDataset
from deepinv.datasets.fastmri import FullMultiCoilFastMRITransform
from deepinv.physics.generator import RandomMaskGenerator

def get_dataset(fnames, mask_generator, img_size, slice_index="middle", target_method="A_dagger"):
    return FastMRISliceDataset(
        "/home/s2558406/RDS/data/fastmri/brain/multicoil_train",
        slice_index=slice_index,
        transform=FullMultiCoilFastMRITransform(
            mask_generator=mask_generator,
            crop_size=img_size,
            target_method=target_method
        ),
        filter_id=lambda s: Path(s.fname).stem in fnames, #and s.metadata["coils"] == n_coils
        load_metadata_from_cache=True,
        save_metadata_to_cache=True,
        metadata_cache_file="/home/s2558406/RDS/data/fastmri/brain/multicoil_train_cache_new.pkl"
    )

if __name__ == "__main__":
    # TODO insert here fname extraction from cache from fastmri-experiments

    parser = ArgumentParser()
    parser.add_argument("--acc", type=int, default=6)
    parser.add_argument("--img_size", type=tuple, default=(384, 320))
    parser.add_argument("--center_fraction", type=float, default=0.0625) # = acs / W
    parser.add_argument("--target_method", type=str, default="A_dagger", choices=("A_adjoint", "A_dagger", "rss"))
    args = parser.parse_args()

    train_fnames = ['file_brain_AXT2_206_2060079', 'file_brain_AXT2_208_2080601', 'file_brain_AXT2_210_6001651', 'file_brain_AXT2_209_2090030', 'file_brain_AXT2_200_6002486', 'file_brain_AXT2_210_6001534', 'file_brain_AXT2_209_2090077', 'file_brain_AXT2_206_2060003', 'file_brain_AXT2_200_2000485', 'file_brain_AXT2_210_2100365', 'file_brain_AXT2_207_2070777', 'file_brain_AXT2_210_2100306', 'file_brain_AXT2_208_2080598', 'file_brain_AXT2_209_6001425', 'file_brain_AXT2_201_2010323', 'file_brain_AXT2_208_2080611', 'file_brain_AXT2_209_6001248', 'file_brain_AXT2_210_6001709', 'file_brain_AXT2_207_2070484', 'file_brain_AXT2_207_2070562', 'file_brain_AXT2_204_2040086', 'file_brain_AXT2_200_2000020', 'file_brain_AXT2_210_2100083', 'file_brain_AXT2_209_6001432', 'file_brain_AXT2_210_2100086', 'file_brain_AXT2_202_2020427', 'file_brain_AXT2_206_2060013', 'file_brain_AXT2_207_2070417', 'file_brain_AXT2_201_2010398', 'file_brain_AXT2_207_2070822', 'file_brain_AXT2_202_2020409', 'file_brain_AXT2_200_6002500', 'file_brain_AXT2_206_2060070', 'file_brain_AXT2_203_2030096', 'file_brain_AXT2_206_2060029', 'file_brain_AXT2_202_2020022', 'file_brain_AXT2_200_2000621', 'file_brain_AXT2_210_6001944', 'file_brain_AXT2_201_2010029', 'file_brain_AXT2_207_2070039', 'file_brain_AXT2_201_2010348', 'file_brain_AXT2_200_6002524', 'file_brain_AXT2_207_2070458', 'file_brain_AXT2_205_6000022', 'file_brain_AXT2_209_6001459', 'file_brain_AXT2_210_6001677', 'file_brain_AXT2_208_2080228', 'file_brain_AXT2_200_2000362', 'file_brain_AXT2_202_2020183', 'file_brain_AXT2_200_6002375', 'file_brain_AXT2_210_6001763', 'file_brain_AXT2_202_2020290', 'file_brain_AXT2_201_2010625', 'file_brain_AXT2_210_6001711', 'file_brain_AXT2_209_2090215', 'file_brain_AXT2_209_6000981', 'file_brain_AXT2_210_6001532', 'file_brain_AXT2_205_2050244', 'file_brain_AXT2_200_6002261', 'file_brain_AXT2_201_2010585', 'file_brain_AXT2_208_2080186', 'file_brain_AXT2_200_2000417', 'file_brain_AXT2_207_2070581', 'file_brain_AXT2_209_2090358', 'file_brain_AXT2_208_2080414', 'file_brain_AXT2_208_2080713', 'file_brain_AXT2_202_2020065', 'file_brain_AXT2_209_6001025', 'file_brain_AXT2_208_2080532', 'file_brain_AXT2_204_2040042', 'file_brain_AXT2_210_6001863', 'file_brain_AXT2_201_2010168', 'file_brain_AXT2_200_6002100', 'file_brain_AXT2_210_6001518', 'file_brain_AXT2_210_6001617', 'file_brain_AXT2_203_2030350', 'file_brain_AXT2_208_2080277', 'file_brain_AXT2_209_6001328', 'file_brain_AXT2_207_2070834', 'file_brain_AXT2_202_2020358', 'file_brain_AXT2_200_2000360', 'file_brain_AXT2_200_2000057', 'file_brain_AXT2_208_2080050', 'file_brain_AXT2_207_2070590', 'file_brain_AXT2_210_2100064', 'file_brain_AXT2_202_2020500', 'file_brain_AXT2_207_2070678', 'file_brain_AXT2_201_2010590', 'file_brain_AXT2_207_2070690', 'file_brain_AXT2_200_6001980', 'file_brain_AXT2_200_2000175', 'file_brain_AXT2_210_2100075', 'file_brain_AXT2_208_2080192', 'file_brain_AXT2_200_6002465', 'file_brain_AXT2_209_2090339', 'file_brain_AXT2_207_2070280', 'file_brain_AXT2_202_2020261', 'file_brain_AXT2_200_6002012', 'file_brain_AXT2_210_6001909', 'file_brain_AXT2_200_6002197', 'file_brain_AXT2_200_2000600', 'file_brain_AXT2_207_2070692', 'file_brain_AXT2_201_2010532', 'file_brain_AXT2_208_2080445', 'file_brain_AXT2_210_6001583', 'file_brain_AXT2_200_2000488', 'file_brain_AXT2_203_2030122', 'file_brain_AXT2_201_2010338', 'file_brain_AXT2_203_2030352', 'file_brain_AXT2_200_6002302', 'file_brain_AXT2_208_2080616', 'file_brain_AXT2_200_6002308', 'file_brain_AXT2_200_2000080', 'file_brain_AXT2_201_2010261', 'file_brain_AXT2_208_2080013', 'file_brain_AXT2_208_2080630', 'file_brain_AXT2_202_6000459', 'file_brain_AXT2_210_6001546', 'file_brain_AXT2_207_2070422', 'file_brain_AXT2_201_2010239', 'file_brain_AXT2_200_6002228', 'file_brain_AXT2_210_6001622', 'file_brain_AXT2_208_2080354', 'file_brain_AXT2_210_6001928', 'file_brain_AXT2_206_2060058', 'file_brain_AXT2_207_2070460', 'file_brain_AXT2_210_2100343', 'file_brain_AXT2_210_6001747', 'file_brain_AXT2_208_2080099', 'file_brain_AXT2_202_2020214', 'file_brain_AXT2_201_2010626', 'file_brain_AXT2_201_2010628', 'file_brain_AXT2_210_6001737', 'file_brain_AXT2_210_2100105', 'file_brain_AXT2_210_6001833', 'file_brain_AXT2_208_2080505', 'file_brain_AXT2_200_6002387', 'file_brain_AXT2_200_6002446', 'file_brain_AXT2_200_6002319', 'file_brain_AXT2_207_2070090', 'file_brain_AXT2_201_2010241', 'file_brain_AXT2_210_6001694', 'file_brain_AXT2_209_6001384', 'file_brain_AXT2_209_6001228', 'file_brain_AXT2_200_6002539', 'file_brain_AXT2_210_2100334', 'file_brain_AXT2_209_6001436', 'file_brain_AXT2_210_2100311', 'file_brain_AXT2_202_2020489', 'file_brain_AXT2_209_2090406', 'file_brain_AXT2_208_2080126', 'file_brain_AXT2_201_2010571', 'file_brain_AXT2_209_2090151', 'file_brain_AXT2_207_2070198', 'file_brain_AXT2_200_6002601', 'file_brain_AXT2_210_2100288', 'file_brain_AXT2_200_6002532', 'file_brain_AXT2_200_2000407', 'file_brain_AXT2_210_2100368', 'file_brain_AXT2_200_6001972', 'file_brain_AXT2_207_2070194', 'file_brain_AXT2_201_2010177', 'file_brain_AXT2_202_2020273', 'file_brain_AXT2_207_2070325', 'file_brain_AXT2_200_6002381', 'file_brain_AXT2_207_2070352', 'file_brain_AXT2_200_2000092', 'file_brain_AXT2_208_2080585', 'file_brain_AXT2_200_6002151', 'file_brain_AXT2_200_6002594', 'file_brain_AXT2_209_6001317', 'file_brain_AXT2_202_2020127', 'file_brain_AXT2_209_6001261', 'file_brain_AXT2_203_2030135', 'file_brain_AXT2_210_6001506', 'file_brain_AXT2_201_2010332', 'file_brain_AXT2_210_6001678', 'file_brain_AXT2_204_2040011', 'file_brain_AXT2_207_2070703', 'file_brain_AXT2_207_2070145', 'file_brain_AXT2_203_2030254', 'file_brain_AXT2_207_2070608', 'file_brain_AXT2_207_2070716', 'file_brain_AXT2_202_2020331', 'file_brain_AXT2_209_6001463', 'file_brain_AXT2_203_2030092', 'file_brain_AXT2_203_2030354', 'file_brain_AXT2_207_2070611', 'file_brain_AXT2_210_2100260', 'file_brain_AXT2_208_2080526', 'file_brain_AXT2_200_6002242', 'file_brain_AXT2_201_2010132', 'file_brain_AXT2_208_2080051', 'file_brain_AXT2_209_2090308', 'file_brain_AXT2_209_6001050', 'file_brain_AXT2_205_2050080', 'file_brain_AXT2_202_2020189', 'file_brain_AXT2_201_2010014', 'file_brain_AXT2_203_2030288', 'file_brain_AXT2_201_6002681', 'file_brain_AXT2_202_2020297', 'file_brain_AXT2_208_2080011', 'file_brain_AXT2_200_6002530', 'file_brain_AXT2_205_2050062', 'file_brain_AXT2_201_2010455', 'file_brain_AXT2_200_2000332', 'file_brain_AXT2_207_2070100', 'file_brain_AXT2_202_2020272', 'file_brain_AXT2_208_2080412', 'file_brain_AXT2_207_2070249', 'file_brain_AXT2_200_6002545', 'file_brain_AXT2_202_2020067', 'file_brain_AXT2_200_2000486', 'file_brain_AXT2_210_6001941', 'file_brain_AXT2_200_6002551', 'file_brain_AXT2_207_2070770', 'file_brain_AXT2_208_2080091', 'file_brain_AXT2_201_2010602', 'file_brain_AXT2_201_2010156', 'file_brain_AXT2_200_2000290', 'file_brain_AXT2_200_2000357', 'file_brain_AXT2_208_2080635', 'file_brain_AXT2_210_6001491', 'file_brain_AXT2_210_2100201']
    test_fnames = ['file_brain_AXT2_200_6002655', 'file_brain_AXT2_200_2000566', 'file_brain_AXT2_202_2020150', 'file_brain_AXT2_201_2010604', 'file_brain_AXT2_208_2080303', 'file_brain_AXT2_201_2010174', 'file_brain_AXT2_208_2080591', 'file_brain_AXT2_204_2040009', 'file_brain_AXT2_206_2060027', 'file_brain_AXT2_207_2070208', 'file_brain_AXT2_209_2090266', 'file_brain_AXT2_200_6002262', 'file_brain_AXT2_207_2070709', 'file_brain_AXT2_210_2100225', 'file_brain_AXT2_201_2010588', 'file_brain_AXT2_200_6002111', 'file_brain_AXT2_200_6002257', 'file_brain_AXT2_200_6002528', 'file_brain_AXT2_203_2030256', 'file_brain_AXT2_200_2000592', 'file_brain_AXT2_210_2100090', 'file_brain_AXT2_200_6002158', 'file_brain_AXT2_205_6000091', 'file_brain_AXT2_201_2010152', 'file_brain_AXT2_202_2020116', 'file_brain_AXT2_205_2050130', 'file_brain_AXT2_207_2070165', 'file_brain_AXT2_210_2100136', 'file_brain_AXT2_202_2020555', 'file_brain_AXT2_200_6002116', 'file_brain_AXT2_202_2020016', 'file_brain_AXT2_210_2100276', 'file_brain_AXT2_204_2040073', 'file_brain_AXT2_210_2100079', 'file_brain_AXT2_202_2020365', 'file_brain_AXT2_210_2100135', 'file_brain_AXT2_208_2080333', 'file_brain_AXT2_210_2100345', 'file_brain_AXT2_208_2080329', 'file_brain_AXT2_200_6002037', 'file_brain_AXT2_209_6000982', 'file_brain_AXT2_202_2020039', 'file_brain_AXT2_209_2090333', 'file_brain_AXT2_209_6001420', 'file_brain_AXT2_208_2080475', 'file_brain_AXT2_202_2020519', 'file_brain_AXT2_210_6001524', 'file_brain_AXT2_201_2010416', 'file_brain_AXT2_207_2070500', 'file_brain_AXT2_203_2030196', 'file_brain_AXT2_208_2080722', 'file_brain_AXT2_202_2020063', 'file_brain_AXT2_207_2070054', 'file_brain_AXT2_201_2010201', 'file_brain_AXT2_210_6001906', 'file_brain_AXT2_210_6001515', 'file_brain_AXT2_201_2010519']

    assert len(train_fnames) + len(test_fnames) == 281

    rng_cpu = torch.Generator().manual_seed(1)

    mask_generator = RandomMaskGenerator(
        img_size=args.img_size,
        acceleration=args.acc,
        center_fraction=args.center_fraction,
        rng=rng_cpu
    )
    assert mask_generator.n_center == 20
    
    rss = "_rss" if args.target_method == "rss" else ""

    train_dataset = get_dataset(train_fnames, mask_generator, args.img_size, slice_index="middle", target_method=args.target_method)
    print(len(train_dataset)) #66*3
    train_dataset.save_local_dataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_train{rss}")

    test_dataset  = get_dataset(test_fnames,  mask_generator, args.img_size, slice_index="middle", target_method=args.target_method)
    print(len(test_dataset)) #83-66=17
    test_dataset.save_local_dataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_test{rss}")

# python scripts/generate_fastmri_multicoil_local.py --acc 6