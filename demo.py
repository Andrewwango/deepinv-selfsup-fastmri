
# %%
import deepinv as dinv
import torch
from torchvision.transforms import Resize

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)
results = {}

# %%
# Define MRI physics with random masks
physics_generator = dinv.physics.generator.GaussianMaskGenerator(
    img_size=(128, 128), acceleration=4, rng=rng, device=device
)
physics = dinv.physics.MRI(img_size=(128, 128), device=device)

# %%
# Define unrolled network


denoiser = dinv.models.DnCNN(
            in_channels=2,  # real + imaginary parts
            out_channels=2,
            pretrained=None,
            depth=15,
        )

denoiser = dinv.models.UNet(
            in_channels=2,
            out_channels=2,
            residual=True, #ResUNet
            batch_norm=True,
            scales=3,
        )

model = lambda: dinv.utils.demo.demo_mri_model(denoiser=denoiser, num_iter=3, device=device).to(device)

# %%
# Define FastMRI datasets
train_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data",
    file_name="fastmri_knee_singlecoil.pt",
    transform=Resize(128),
    train=True,
    train_percent=0.8,
    download=True,
)

test_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data",
    file_name="fastmri_knee_singlecoil.pt",
    transform=Resize(128),
    train=False,
    train_percent=0.8,
)

# Simulate and save random measurements
dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    physics_generator=physics_generator,
    save_physics_generator_params=True,
    overwrite_existing=False,
    device=device,
    save_dir="data",
    batch_size=1,
)

# Load saved datasets
train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

train_dataloader, test_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True), torch.utils.data.DataLoader(test_dataset)

# %%
from trainer import Trainer
def train(loss: dinv.loss.Loss, epochs: int = 0):
    _model = model()
    
    trainer = Trainer(
        model = _model,
        physics = physics,
        optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3),
        train_dataloader = train_dataloader,
        eval_dataloader = test_dataloader,
        epochs = epochs,
        losses = loss,
        scheduler = None,
        metrics = dinv.metric.PSNR(complex_abs=True),
        ckp_interval = 10,
        device = device,
        eval_interval = 1,
        save_path = f"/home/s2558406/RDS/models/deepinv-selfsup-fastmri/{args.loss}",
        plot_images = False,
        wandb_vis = True,
        log_train_batch=True,
    )

    trainer.train()
    trainer.plot_images = True
    trainer.wandb_vis = False
    return trainer

# %%
from argparse import ArgumentParser
import json
parser = ArgumentParser()
parser.add_argument("--loss", type=str, default="ei")
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()
match args.loss:
    case "temp":
        loss = dinv.loss.SupLoss()
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
    case "mo-ei":
        from moi import MOEILoss
        loss = [
            dinv.loss.MCLoss(),
            MOEILoss(transform=dinv.transform.CPABDiffeomorphism(device=device), physics_generator=physics_generator)
        ]
    case "ssdu":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 128, 128), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=False
        )
    case "noise2inverse":
        loss = dinv.loss.SplittingLoss(
            mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2, 128, 128), split_ratio=0.6, device=device, rng=rng),
            eval_split_input=True, eval_n_samples=5
        )
    case "noisier2noise-ssdu":
        loss = ...

# Set epochs > 0 to train the model
import wandb
with wandb.init(project="deepinv-selfsup-fastmri-experiments", config={"loss": args.loss}):
    trainer = train(loss, epochs=args.epochs)
results = trainer.test(test_dataloader)

with open(f"/home/s2558406/RDS/models/deepinv-selfsup-fastmri/{args.loss}/results.json", "w") as f:
    json.dump(results, f)