
# %%
import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

results = {}

model_dir = "/home/s2558406/RDS/models/deepinv-selfsup-fastmri"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--loss", type=str, default="ei")
parser.add_argument("--x_metric", type=str, default="mse", choices=("mse", "ssim-mse"))
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--physics", type=str, default="mri", choices=("mri", "noisy", "multicoil"))
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
parser.add_argument("-b", type=int, default=4)
args = parser.parse_args()

torch.manual_seed(args.global_seed)
torch.cuda.manual_seed(args.global_seed)
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)

# %%
# Define MRI physics with random masks
physics_generator = dinv.physics.generator.GaussianMaskGenerator(
    img_size=(320, 320), acceleration=args.acc, rng=rng, device=device
)
physics = dinv.physics.MRI(img_size=(320, 320), device=device)

if args.physics == "noisy":
    physics.noise_model = dinv.physics.GaussianNoise(0.1, rng=rng)
elif args.physics == "multicoil":
    physics = dinv.physics.MultiCoilMRI(img_size=(320, 320), coil_maps=8, device=device)

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
    physics_generator=physics_generator,
    save_physics_generator_params=True,
    overwrite_existing=False,
    device=device,
    save_dir=model_dir.replace("models", "data"),
    batch_size=1,
    dataset_filename="dinv_dataset_paper" + (f"_{args.data}" if args.data != "knee" else "") + (f"_{args.acc}" if args.acc != 8 else "") + ("_noisy" if args.physics == "noisy" else "") + ("_multicoil" if args.physics == "multicoil" else "")
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
        print("ADVERSARIAL TRAINING")
        optimizer = dinv.training.AdversarialOptimizer(optimizer, torch.optim.Adam(discrim.parameters(), args.lr if args.lr is not None else 1e-3))
        scheduler = dinv.training.adversarial.AdversarialScheduler(scheduler, torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule) if args.schedule is not None else None)
        _trainer = dinv.training.AdversarialTrainer
    else:
        _trainer = dinv.Trainer

    class PSNR2(dinv.metric.PSNR):
        def __init__(self, **kwargs):
            super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
    class SSIM2(dinv.metric.SSIM):
        def __init__(self, **kwargs):
            super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
    class PSNR3(dinv.metric.PSNR):
        def metric(self, x_net, x, *args, **kwargs):
            return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)
    class SSIM3(dinv.metric.SSIM):
        def metric(self, x_net, x, *args, **kwargs):
            return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)           

    trainer = _trainer(
        model = _model,
        physics = physics,
        optimizer = optimizer,
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = epochs,
        losses = loss,
        scheduler = scheduler,
        metrics = [dinv.metric.PSNR(complex_abs=True), dinv.metric.SSIM(complex_abs=True)] + ([PSNR2(), SSIM2(), PSNR3(complex_abs=True), SSIM3(complex_abs=True)] if args.norm_metrics else []),
        ckp_interval = 10,
        device = device,
        eval_interval = 1,
        save_path = None,
        plot_images = False,
        wandb_vis = True,
        ckpt_pretrained=None if args.ckpt is None else f"{model_dir}/paper/{args.ckpt}"
    )

    if discrim is not None:
        trainer.D = discrim
        trainer.losses_d = loss_d
        trainer.step_ratio_D = 1 #sync G and D

    if args.ckpt is not None and args.lr is not None:
        for param in trainer.optimizer.param_groups:
            param["lr"] = args.lr
        print(trainer.optimizer.param_groups[0]["lr"])

    trainer.train()
    trainer.plot_images = True
    trainer.wandb_vis = False
    trainer.ckpt_pretrained = None
    return trainer

# %%
class SumMetric(dinv.metric.Metric):
    def __init__(self, *metrics: dinv.metric.Metric):
        super().__init__()
        self.metrics = metrics
    def forward(self, x_net = None, x = None, *args, **kwargs):
        return sum([m.forward(x_net, x, *args, **kwargs) for m in self.metrics])

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

match args.loss:
    case "mc":
        loss = dinv.loss.MCLoss()
    case "sup":
        loss = dinv.loss.SupLoss(metric=xm)
    case "ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.EILoss(transform=rotate, metric=xm)
        ]
    case "diffeo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.EILoss(transform=diffeo, metric=xm)
        ]
    case "moi":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOILoss(physics_generator=physics_generator, metric=xm)
        ]
    case "rotate-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=rotate, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=diffeo, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo-moi-ei":
        from utils.loss_scheduler import RandomLossScheduler
        loss = [
            dinv.loss.MCLoss(),
            RandomLossScheduler(
                dinv.loss.MOILoss(physics_generator=physics_generator, metric=xm),
                dinv.loss.EILoss(transform=diffeo, metric=xm),
            )
        ]
    case "diffeo-rotate-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=diffeo_rotate, physics_generator=physics_generator, metric=xm)
        ]
    case "diffeo*rotate-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=diffeo_rotate2, physics_generator=physics_generator, metric=xm)
        ]
    case "sup-diffeo-mo-ei":
        from utils.loss_scheduler import RandomLossScheduler
        loss = RandomLossScheduler(
            dinv.loss.SupLoss(),
            [
                dinv.loss.MCLoss(),
                dinv.loss.MOEILoss(transform=diffeo, physics_generator=physics_generator, metric=xm)
            ],
            generator=None,
            weightings=[3, 1]
        )
    case "ssdu":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 320, 320), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )
    case "noise2inverse":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 320, 320), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=3
        )
    case "noisier2noise-ssdu":
        loss = ...
    case "ei-sure":
        loss = [
            dinv.loss.SureGaussianLoss(sigma=0.),
            dinv.loss.MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator, metric=xm)
        ]
    case "cole":
        from utils.multi_operator_adversarial_consistency import MultiOperatorUnsupAdversarialDiscriminatorLoss, MultiOperatorUnsupAdversarialGeneratorLoss, SkipConvDiscriminator
        discrim = SkipConvDiscriminator((320, 320)).to(device)
        
        dataloader_factory = lambda: torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, generator=torch.Generator("cpu").manual_seed(42))
        physics_generator_factory = lambda: dinv.physics.generator.GaussianMaskGenerator(img_size=(320, 320), acceleration=args.acc, rng=torch.Generator(device).manual_seed(42), device=device)
        
        loss = MultiOperatorUnsupAdversarialGeneratorLoss(D=discrim, device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory)
        loss_d=MultiOperatorUnsupAdversarialDiscriminatorLoss(D=discrim, device=device, dataloader_factory=dataloader_factory, physics_generator_factory=physics_generator_factory)
    case "vortex":
        from deepinv.transform import Rotate, Shift, Scale, Reflect
        from deepinv.transform.projective import Affine
        from utils.vortex import VORTEXLoss, NoiseTransform, RandomPhaseShift
        loss = VORTEXLoss(
            Shift(shift_max=0.1, rng=rng),# | Scale(factors=[0.75, 1.25], rng=rng) | Rotate(rng=rng) | Reflect(rng=rng),
            RandomPhaseShift(scale=0.4, rng=rng),#NoiseTransform(rng=rng)
        )
        loss = [dinv.loss.MCLoss(), loss]
        # Affine(theta_z_max=0, zoom_factor_min=1., shift_max=0., skew_max=10, x_stretch_factor_min=1., y_stretch_factor_min=1., device=device)

# Set epochs > 0 to train the model
import wandb, json
with wandb.init(project="deepinv-selfsup-fastmri-experiments", config={"loss": args.loss}):
    run_id = wandb.run.id
    trainer = train(loss, epochs=args.epochs, loss_d=loss_d, discrim=discrim)
    trainer.save_folder_im = f"{model_dir}/paper/{run_id}"

if isinstance(loss, dinv.loss.SplittingLoss):
    assert isinstance(trainer.model, dinv.loss.SplittingLoss.SplittingModel)
    assert isinstance(trainer.model.mask_generator, dinv.physics.generator.GaussianSplittingMaskGenerator)
    if args.loss == "noise2inverse":
        assert trainer.model.eval_split_input == True
    print("Success")

if args.save_model:
    trainer.save_path = f"{model_dir}/paper/{run_id}"
    trainer.save_model(trainer.epochs - 1)

results = trainer.test(test_dataloader, f"{model_dir}/paper/{run_id}")
with open(f"{model_dir}/paper/{run_id}/results.json", "w") as f:
    json.dump(results, f)

if not args.no_save:
    sample_xhat, sample_x, sample_y, sample_xinit = [], [], [], []
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

    from numpy import savez
    samples_to_save = {
        "x_hat": torch.cat(sample_xhat).detach().cpu().numpy()
    }

    if args.save_gt:
        samples_to_save |= {
            "x": torch.cat(sample_x).detach().cpu().numpy(),
            "y": torch.cat(sample_y).detach().cpu().numpy(),
            "x_init": torch.cat(sample_xinit).detach().cpu().numpy()
        }

    savez(f"{model_dir}/paper/{run_id}/samples.npz", **samples_to_save)

# python train_paper.py --loss "sup" --epochs 150 --save_model --schedule 20 --save_gt --acc 6
# python train_paper.py --loss "ssdu" --epochs 150 --save_model --acc 6
# python train_paper.py --loss "noise2inverse" --epochs 0 --ckpt "i65an1aa/ckpt_149.pth.tar"
# python train_paper.py --loss "sup" --epochs 150 --save_model --schedule 20 --save_gt --data "brain" --acc 6