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
    def __getitem__(self, idx):
        x, _, params = super().__getitem__(idx)
        params = {"mask" : params["mask"]} #discard coil_maps
        physics = MultiCoilMRI(img_size=x.shape[-2:], coil_maps=None, **params)
        y = physics(x.unsqueeze(0)).squeeze(0)
        params["coil_maps"] = physics.coil_maps
        return x, y, params