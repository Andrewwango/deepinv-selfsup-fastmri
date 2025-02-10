
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
train_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data",
    file_name=file_name,
    train=True,
    train_percent=0.8,
    download=True,
)

test_dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data",
    file_name="fastmri_knee_singlecoil.pt",
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
    save_dir=model_dir.replace("models", "data"),
    batch_size=1,
    dataset_filename="dinv_dataset_paper" + ("_noisy" if args.physics == "noisy" else "") + ("_multicoil" if args.physics == "multicoil" else "")
)

# Load saved datasets
train_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="train", load_physics_generator_params=True)
test_dataset = dinv.datasets.HDF5Dataset(dataset_path, split="test", load_physics_generator_params=True)

train_dataloader, test_dataloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, [0]), shuffle=True, generator=rng_cpu), torch.utils.data.DataLoader(test_dataset)

# %%
def train(loss: dinv.loss.Loss, epochs: int = 0):
    _model = model()
    
    trainer = dinv.Trainer(
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
        save_path = None,
        plot_images = False,
        wandb_vis = True,
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
    case "mo-ei":
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
            eval_split_input=True, eval_n_samples=5
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
results["train"] = trainer.test(train_dataloader, save_path=None)

samples = []
iterator = iter(test_dataloader)
for _ in range(5):
    x, y, params = next(iterator)
    params = {k: v.to(device) for (k, v) in params.items()}
    physics.update_parameters(**params)
    samples += [trainer.model(y.to(device), physics)]

with open(f"{model_dir}/paper/{run_id}/results.json", "w") as f:
    json.dump(results, f)

from numpy import save
save(f"{model_dir}/paper/{run_id}/samples.npy", torch.cat(samples).detach().cpu().numpy())

# python train_paper.py --loss "sup" --epochs 0