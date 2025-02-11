
# %%
import deepinv as dinv
import torch
from torchvision.transforms import Resize

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)
results = {}

file_name = "fastmri_knee_singlecoil.pt"
model_dir = "/home/s2558406/RDS/models/deepinv-selfsup-fastmri"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--loss", type=str, default="ei")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--physics", type=str, default="mri", choices=("mri", "noisy", "multicoil"))
parser.add_argument("--save_gt", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--scheduler", action="store_true")
parser.add_argument("--ckpt", type=str, default=None)
args = parser.parse_args()

# %%
# Define MRI physics with random masks
physics_generator = dinv.physics.generator.GaussianMaskGenerator(
    img_size=(320, 320), acceleration=8, rng=rng, device=device
)
physics = dinv.physics.MRI(img_size=(320, 320), device=device)

if args.physics == "noisy":
    physics.noise_model = dinv.physics.GaussianNoise(0.1, rng=rng)
elif args.physics == "multicoil":
    physics = dinv.physics.MultiCoilMRI(img_size=(320, 320), coil_maps=8, device=device)

# %%
# Define unrolled network
denoiser = dinv.models.UNet(2, 2, scales=4, batch_norm=False)
model = lambda: dinv.utils.demo.demo_mri_model(denoiser=denoiser, num_iter=3, device=device).to(device)

# %%
# Define FastMRI datasets
dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data",
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
    dataset_filename="dinv_dataset_paper" + ("_noisy" if args.physics == "noisy" else "") + ("_multicoil" if args.physics == "multicoil" else "")
)

# Load saved datasets
train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

train_dataloader, test_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, generator=rng_cpu, batch_size=4), torch.utils.data.DataLoader(test_dataset, batch_size=4)

# %%
def train(loss: dinv.loss.Loss, epochs: int = 0):
    _model = model()
    optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
    trainer = dinv.Trainer(
        model = _model,
        physics = physics,
        optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3),
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = epochs,
        losses = loss,
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50]) if args.scheduler else None,
        metrics = [dinv.metric.PSNR(complex_abs=True), dinv.metric.SSIM(complex_abs=True)],
        ckp_interval = 10,
        device = device,
        eval_interval = 1,
        save_path = None,
        plot_images = False,
        wandb_vis = True,
        ckpt_pretrained=None if args.ckpt is None else f"{model_dir}/paper/{args.ckpt}"
    )

    trainer.train()
    trainer.plot_images = True
    trainer.wandb_vis = False
    return trainer

# %%
match args.loss:
    case "mc":
        loss = dinv.loss.MCLoss()
    case "sup":
        loss = dinv.loss.SupLoss()
    case "ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.EILoss(transform=dinv.transform.Rotate())
        ]
    case "diffeo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.EILoss(transform=dinv.transform.CPABDiffeomorphism(device=device))
        ]
    case "moi":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOILoss(physics_generator=physics_generator)
        ]
    case "rotate-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=dinv.transform.Rotate(), physics_generator=physics_generator)
        ]
    case "diffeo-mo-ei":
        loss = [
            dinv.loss.MCLoss(),
            dinv.loss.MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator)
        ]
    case "ssdu":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 128, 128), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )
    case "noise2inverse":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 128, 128), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=10
        )
    case "noisier2noise-ssdu":
        loss = ...
    case "ei-sure":
        loss = [
            dinv.loss.SureGaussianLoss(sigma=0.),
            dinv.loss.MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator)
        ]

# Set epochs > 0 to train the model
import wandb, json
with wandb.init(project="deepinv-selfsup-fastmri-experiments", config={"loss": args.loss}):
    run_id = wandb.run.id
    trainer = train(loss, epochs=args.epochs)
    trainer.save_folder_im = f"{model_dir}/paper/{run_id}"

results = trainer.test(test_dataloader, f"{model_dir}/paper/{run_id}")

with open(f"{model_dir}/paper/{run_id}/results.json", "w") as f:
    json.dump(results, f)

if args.save_model:
    trainer.save_path = f"{model_dir}/paper/{run_id}"
    trainer.save_model(trainer.epochs - 1)

sample_xhat, sample_x, sample_y, sample_xinit = [], [], [], []
iterator = iter(test_dataloader)
for _ in range(5):
    x, y, params = next(iterator)
    params = {k: v.to(device) for (k, v) in params.items()}
    physics.update_parameters(**params)
    sample_xhat += [trainer.model(y.to(device), physics)]
    sample_x += [x]
    sample_y += [y]
    sample_xinit += [physics.A_adjoint(y.to(device), **params)]

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

# python train_paper.py --loss "sup" --epochs 150 --scheduler --save_gt
# python train_paper.py --loss "ssdu" --epochs 150 --save_model
# python train_paper.py --loss "noise2inverse" --epochs 0 --ckpt "i65an1aa/ckpt_149.pth.tar"