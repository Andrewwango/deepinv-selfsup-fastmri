import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%
import deepinv as dinv
import torch
from utils import *

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

results = {}

model_dir = "/home/s2558406/RDS/models/deepinv-selfsup-fastmri"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--loss", type=str, default="ei")
parser.add_argument("-b", type=int, default=4, help="batch size")
parser.add_argument("--x_metric", type=str, default="mse", choices=("mse", "ssim-mse"))
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--physics", type=str, default="mri", choices=("mri", "noisy", "multicoil", "single"), help="Default multi-operator singlecoil MRI, Gaussian noised, Multicoil, or single-operator")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--save_gt", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--schedule", type=int, default=None, nargs="+")
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--acc", type=int, default=6)
parser.add_argument("--data", type=str, default="knee", choices=("knee", "brain"))
parser.add_argument("-lr", type=float, default=None)
parser.add_argument("--global_seed", type=int, default=0)
parser.add_argument("--model", type=str, default="modl", choices=("modl", "varnet"))
parser.add_argument("--unroll", type=int, default=3)
parser.add_argument("--norm_metrics", action="store_true")
parser.add_argument("--n_coils", type=int, default=16)
parser.add_argument("--compare_dagger", action="store_true")
parser.add_argument("--adj_mc", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.global_seed)
torch.cuda.manual_seed(args.global_seed)
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)

# %%
# Define MRI physics with random masks
img_size = (320, 320)
if args.physics == "multicoil":
    img_size = (384, 320)

physics_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=args.acc, rng=rng, device=device)
physics = dinv.physics.MRI(img_size=img_size, device=device)

match args.physics:
    case "noisy":
        sigma = 0.1
        physics.noise_model = dinv.physics.GaussianNoise(sigma, rng=rng)
    case "multicoil":
        physics = dinv.physics.MultiCoilMRI(device=device)
        physics_generator = dinv.physics.generator.RandomMaskGenerator(img_size=img_size, acceleration=args.acc, center_fraction=0.0625, rng=rng, device=device)
    case "single":
        physics.update(**physics_generator.step())

# %%
# Define unrolled network
denoiser = dinv.models.UNet(2, 2, scales=4, batch_norm=False)
match args.model:
    case "modl":
        model = lambda: dinv.utils.demo.demo_mri_model(denoiser=denoiser, num_iter=args.unroll, device=device).to(device)
    case "varnet":
        model = lambda: dinv.models.VarNet(denoiser, num_cascades=args.unroll)

# %%
# Define FastMRI datasets
if args.physics == "multicoil":
    if args.data != "brain":
        raise ValueError("data must be brain for multicoil.")

    train_dataset = dinv.datasets.LocalDataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_train")
    test_dataset  = dinv.datasets.LocalDataset(f"/home/s2558406/RDS/data/fastmri/brain/multicoil_train_slices_{args.acc}_{args.n_coils}_test")
else:
    match args.data:
        case "knee":
            file_name = "fastmri_knee_singlecoil.pt"
        case "brain":
            file_name = "fastmri_brain_singlecoil.pt"

    dataset = dinv.datasets.SimpleFastMRISliceDataset(
        model_dir.replace("models", "data"),
        file_name=file_name,
        train=True,
        train_percent=1.,
        download=True,
    )
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2), generator=rng_cpu)

    # Simulate and save random measurements
    dataset_path = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        physics_generator=physics_generator if args.physics != "single" else None,
        save_physics_generator_params=True,
        overwrite_existing=False,
        device=device,
        save_dir=model_dir.replace("models", "data"),
        batch_size=1,
        dataset_filename="dinv_dataset_paper" + (f"_{args.data}" if args.data != "knee" else "") + (f"_{args.acc}" if args.acc != 8 else "") + ("_noisy" if args.physics == "noisy" else "") + ("_single" if args.physics == "single" else "")
    )

    # Load saved datasets
    train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
    test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

train_dataloader, test_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, generator=rng_cpu, batch_size=args.b), torch.utils.data.DataLoader(test_dataset, batch_size=args.b)

# %%
def train(loss: dinv.loss.Loss, epochs: int = 0, discrim: torch.nn.Module=None, loss_d: dinv.loss.adversarial.DiscriminatorLoss = None):
    _model = model()
    optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr if args.lr is not None else 1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule) if args.schedule is not None else None

    if discrim is not None:
        optimizer = dinv.training.AdversarialOptimizer(optimizer, torch.optim.Adam(discrim.parameters(), args.lr if args.lr is not None else 1e-3))
        scheduler = dinv.training.adversarial.AdversarialScheduler(scheduler, torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule)) if args.schedule is not None else None
        _trainer = dinv.training.AdversarialTrainer
    else:
        _trainer = dinv.Trainer      

    if args.physics == "multicoil":
        #metrics = [CropPSNR(complex_abs=True, max_pixel=None), CropSSIM(complex_abs=True, max_pixel=None)]
        metrics = [dinv.metric.PSNR(complex_abs=True, max_pixel=None), dinv.metric.SSIM(complex_abs=True, max_pixel=None), dinv.metric.NMSE(complex_abs=True)]
    else:
        metrics = [dinv.metric.PSNR(complex_abs=True), dinv.metric.SSIM(complex_abs=True)]
    if args.norm_metrics:
        metrics += [PSNR2(), SSIM2(), PSNR3(complex_abs=True), SSIM3(complex_abs=True)]

    trainer = _trainer(
        model = _model,
        physics = physics,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = epochs,
        losses = loss,
        scheduler = scheduler,
        metrics = metrics,
        ckp_interval = 10,
        device = device,
        eval_interval = 1,
        save_path = None,
        plot_images = False,
        wandb_vis = True,
        ckpt_pretrained=None if args.ckpt is None else f"{model_dir}/paper/{args.ckpt}",
        no_learning_method="A_dagger" if args.compare_dagger else "A_adjoint",
    )

    if discrim is not None:
        trainer.D = discrim
        trainer.losses_d = loss_d
        trainer.step_ratio_D = 1 #sync G and D

    if args.ckpt is not None and args.lr is not None:
        for param in trainer.optimizer.param_groups:
            param["lr"] = args.lr

    trainer.train()
    trainer.plot_images = True
    trainer.wandb_vis = False
    trainer.ckpt_pretrained = None
    return trainer

# %%
match args.x_metric:
    case "mse":
        xm = torch.nn.MSELoss()
    case "ssim-mse":
        xm = SumMetric(dinv.metric.SSIM(train_loss=True, complex_abs=True, reduction="mean"), torch.nn.MSELoss())

loss_d = None
discrim = None

rotate = dinv.transform.Rotate()
diffeo = dinv.transform.CPABDiffeomorphism(device=device)
diffeo_rotate = rotate | diffeo
diffeo_rotate2 = rotate * diffeo

if args.adj_mc:
    mcloss = AdjMCLoss()
else:
    mcloss = dinv.loss.MCLoss()

match args.loss:
    case "mc":
        loss = mcloss
    case "sup":
        loss = dinv.loss.SupLoss(metric=xm)
    case "ei":
        loss = [
            mcloss,
            dinv.loss.EILoss(transform=rotate, metric=xm)
        ]
    case "diffeo-ei":
        loss = [
            mcloss,
            dinv.loss.EILoss(transform=diffeo, metric=xm)
        ]
    case "moi":
        loss = [
            mcloss,
            dinv.loss.MOILoss(physics_generator=physics_generator, metric=xm)
        ]
    case "rotate-mo-ei":
        loss = [
            mcloss,
            dinv.loss.MOEILoss(transform=rotate, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo-mo-ei":
        loss = [
            mcloss,
            dinv.loss.MOEILoss(transform=diffeo, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo-moi-ei":
        loss = [
            mcloss,
            RandomLossScheduler(
                dinv.loss.MOILoss(physics_generator=physics_generator, metric=xm),
                dinv.loss.EILoss(transform=diffeo, metric=xm),
            )
        ]
    case "diffeo-rotate-mo-ei":
        loss = [
            mcloss,
            dinv.loss.MOEILoss(transform=diffeo_rotate, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo*rotate-mo-ei":
        loss = [
            mcloss,
            dinv.loss.MOEILoss(transform=diffeo_rotate2, physics_generator=physics_generator, metric=xm)
        ]
    case "sup-diffeo-mo-ei":
        loss = RandomLossScheduler(
            dinv.loss.SupLoss(),
            [
                mcloss,
                dinv.loss.MOEILoss(transform=diffeo, physics_generator=physics_generator, metric=xm)
            ],
            generator=None,
            weightings=[3, 1]
        )
    case "ssdu":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )
    case "ssdu-bernoulli":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )        
    case "noise2inverse":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=3
        )
    case "noise2inverse-bernoulli":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=3
        )
    case "weighted-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)
    case "weighted-ssdu-ablation-1":
        mask_generator = dinv.physics.generator.GaussianSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng)
        loss = dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)
    case "weighted-ssdu-ablation-2":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.SplittingLoss(mask_generator=mask_generator, eval_split_input=False)
    case "weighted-ssdu-ablation-3":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.SplittingLoss(mask_generator=mask_generator, eval_split_input=True, eval_n_samples=3)
    case "weighted-ssdu-bernoulli":
        mask_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator((1, *img_size), split_ratio=0.6, device=device, rng=rng)
        loss = dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)
    case "weighted-ssdu-3":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=3, rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)
    case "weighted-ssdu-no-acs":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)        
    case "ssdu-1d-no-acs":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = dinv.loss.SplittingLoss(mask_generator=mask_generator, eval_split_input=False)
    
    case "cole":
        discrim = SkipConvDiscriminator(img_size, use_sigmoid=False).to(device)
        
        dataloader_factory = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, generator=torch.Generator("cpu").manual_seed(42))
        physics_generator_factory = lambda: dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=args.acc, rng=torch.Generator(device).manual_seed(42), device=device)
        
        loss = MultiOperatorUnsupAdversarialGeneratorLoss(device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory)
        loss_d=MultiOperatorUnsupAdversarialDiscriminatorLoss(device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory)

    case "uair":
        discrim = SkipConvDiscriminator(img_size, use_sigmoid=False).to(device)
        physics_generator_factory = lambda: dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=args.acc, rng=torch.Generator(device).manual_seed(42), device=device)
        loss = UAIRGeneratorLoss(device=device, physics_generator_factory=physics_generator_factory)
        loss_d=UAIRDiscriminatorLoss(device=device, physics_generator_factory=physics_generator_factory)
    
    case "vortex":
        loss = [mcloss, VORTEXLoss(rng=rng)]

    case "sure-diffeo-mo-ei":
        loss = [dinv.loss.SureGaussianLoss(sigma=sigma), dinv.loss.MOEILoss(transform=diffeo, physics_generator=physics_generator, metric=xm)]
    
    case "sure-ei":
        loss = [dinv.loss.SureGaussianLoss(sigma=sigma), dinv.loss.EILoss(transform=rotate, metric=xm)]

    case "robust-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = RobustSplittingLoss(mask_generator, physics_generator, dinv.physics.GaussianNoise(sigma=sigma, rng=torch.Generator(device).manual_seed(42)))
    
    case "noise2recon-ssdu":
        split_generator = dinv.physics.generator.GaussianMaskGenerator(img_size=img_size, acceleration=2, center_fraction=0., rng=rng, device=device)
        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((1, *img_size), split_generator, device=device)
        loss = [
            dinv.loss.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator),
            VORTEXLoss(RandomNoise(sigma=(sigma * 0.5, sigma * 2), rng=rng), IdentityTransform(), no_grad=False)
        ]
    
    case "ensure":
        loss = ENSURELoss(sigma=sigma, physics_generator=physics_generator)

import wandb, json
with wandb.init(project="deepinv-selfsup-fastmri-experiments", config=vars(args)):
    run_id = wandb.run.id
    trainer = train(loss, epochs=args.epochs, loss_d=loss_d, discrim=discrim)
    trainer.save_folder_im = f"{model_dir}/paper/{run_id}"

if args.save_model:
    trainer.save_path = f"{model_dir}/paper/{run_id}"
    trainer.save_model(trainer.epochs - 1)

results = trainer.test(test_dataloader, f"{model_dir}/paper/{run_id}", log_raw_metrics=True)
with open(f"{model_dir}/paper/{run_id}/results.json", "w") as f:
    json.dump(results, f)

if not args.no_save:
    sample_xhat, sample_x, sample_y, sample_xinit, sample_xpinv = [], [], [], [], []
    iterator = iter(test_dataloader)
    trainer.model = trainer.model.to("cpu")
    physics.device = "cpu"
    physics = physics.to("cpu")
    for _ in range(5):
        x, y, params = next(iterator)
        physics.update_parameters(**params)
        sample_xhat += [trainer.model(y, physics)]
        sample_x += [x]
        sample_y += [y]
        sample_xinit += [physics.A_adjoint(y, **params)]
        sample_xpinv += [physics.A_dagger(y, **params)]

    def crop_smallest(x):
        if args.physics == "multicoil":
            #sh = (min([xi.shape[-2] for xi in x]), min([xi.shape[-1] for xi in x]))
            #return [MRIMixin().crop(xi, shape=sh) for xi in x]
            return x
        else:
            return x

    from numpy import savez
    samples_to_save = {
        "x_hat": torch.cat(crop_smallest(sample_xhat)).detach().cpu().numpy()
    }

    if args.save_gt:
        samples_to_save |= {
            "x": torch.cat(crop_smallest(sample_x)).detach().cpu().numpy(),
            "y": torch.cat(crop_smallest(sample_y)).detach().cpu().numpy(),
            "x_init": torch.cat(crop_smallest(sample_xinit)).detach().cpu().numpy(),
            "x_pinv": torch.cat(crop_smallest(sample_xpinv)).detach().cpu().numpy()
        }

    savez(f"{model_dir}/paper/{run_id}/samples.npz", **samples_to_save)

# python scripts/train_paper.py --loss "sup" --epochs 150 --save_model --schedule 20 --save_gt --acc 6
# python scripts/train_paper.py --loss "ssdu" --epochs 150 --save_model --acc 6
# python scripts/train_paper.py --loss "noise2inverse" --epochs 0 --ckpt "i65an1aa/ckpt_149.pth.tar"
# python scripts/train_paper.py --loss "sup" --epochs 150 --save_model --schedule 20 --save_gt --data "brain" --acc 6

# python scripts/train_paper.py --loss weighted-ssdu-ablation-1 --data brain --epochs 120 --save_model -lr 1e-4

# python scripts/train_paper.py --physics multicoil --acc 8 --n_coils 16 --data brain -b 4 --loss diffeo-mo-ei --adj_mc -lr 1e-3 --epochs 30 --save_model --save_gt