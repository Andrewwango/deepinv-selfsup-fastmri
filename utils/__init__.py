from .loss_scheduler import RandomLossScheduler
from .multi_operator_adversarial_consistency import MultiOperatorUnsupAdversarialDiscriminatorLoss, MultiOperatorUnsupAdversarialGeneratorLoss
from .vortex import VORTEXLoss, RandomNoise, RandomPhaseError
from .uair import UAIRGeneratorLoss, UAIRDiscriminatorLoss
from .gan import SkipConvDiscriminator
from .robust_ssdu import RobustSplittingLoss
from .identity_transform import IdentityTransform

from deepinv.loss.metric import PSNR, SSIM, Metric

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