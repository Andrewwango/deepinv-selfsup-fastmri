from .loss_scheduler import RandomLossScheduler
from .multi_operator_adversarial_consistency import MultiOperatorUnsupAdversarialDiscriminatorLoss, MultiOperatorUnsupAdversarialGeneratorLoss
from .vortex import VORTEXLoss, RandomNoise, RandomPhaseError
from .uair import UAIRGeneratorLoss, UAIRDiscriminatorLoss
from .gan import SkipConvDiscriminator
from .robust_ssdu import RobustSplittingLoss
from .identity_transform import IdentityTransform
from .ensure import ENSURELoss

from deepinv.loss.metric import PSNR, SSIM, Metric
from deepinv.physics.mri import MRIMixin, MultiCoilMRI, MRI
from deepinv.loss.mc import MCLoss
from deepinv.physics.generator import GaussianMaskGenerator
from deepinv.physics.generator.base import seed_from_string

class SumMetric(Metric):
    def __init__(self, *metrics: Metric):
        super().__init__()
        self.metrics = metrics
    def forward(self, x_net = None, x = None, *args, **kwargs):
        return sum([m.forward(x_net, x, *args, **kwargs) for m in self.metrics])

class PSNR2(PSNR):
    def __init__(self, **kwargs):
        super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
class SSIM2(SSIM):
    def __init__(self, **kwargs):
        super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
class PSNR3(PSNR):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)
class SSIM3(SSIM):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)

class CropSSIM(SSIM, MRIMixin):
    def forward(self, x_net=None, x=None, *args, **kwargs):
        return super().forward(self.crop(x_net, shape=x.shape), x, *args, **kwargs)

class CropPSNR(PSNR, MRIMixin):
    def forward(self, x_net=None, x=None, *args, **kwargs):
        return super().forward(self.crop(x_net, shape=x.shape), x, *args, **kwargs)

class AdjMCLoss(MCLoss):
    def forward(self, y, x_net, physics, **kwargs):
        return self.metric(physics.A_adjoint(physics.A(x_net)), physics.A_adjoint(y))

from deepinv.datasets.fastmri import LocalDataset
class SimulatedLocalDataset(LocalDataset):
    def __init__(self, *args, simulate_coils=0, simulated2=False, physics_generator: GaussianMaskGenerator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulate_coils = simulate_coils
        self.simulated2 = simulated2
        self.physics_generator = physics_generator

    def __getitem__(self, idx):
        x, _, params = super().__getitem__(idx)
        params = {"mask" : params["mask"]} #discard coil_maps
        
        if self.simulated2:
            # crop x
            x = MRIMixin().crop(x, shape=self.physics_generator.img_size[-2:])
            # normalise x
            x = (x - x.min()) / (x.max() - x.min())
            # discard mask
            params["mask"] = self.physics_generator.step(seed=seed_from_string(str(self.files[idx])))["mask"].squeeze(0)

        if self.simulate_coils == 0:
            physics = MultiCoilMRI(img_size=x.shape[-2:], coil_maps=None, **params)
            
            if self.simulated2:
                physics = MRI(img_size=x.shape[-2:], mask=params["mask"])
        else:
            physics = MultiCoilMRI(img_size=x.shape[-2:], coil_maps=self.simulate_coils, **params)

        y = physics(x.unsqueeze(0)).squeeze(0)
        
        if not self.simulated2:
            params["coil_maps"] = physics.coil_maps.squeeze(0)
        
        return x, y, params