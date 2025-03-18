from deepinv.loss.sure import SureGaussianLoss
from deepinv.physics.mri import MRIMixin

from torch import Tensor
import torch

class W(MRIMixin):
    def __init__(self, physics_generator):
        super().__init__()
        D = physics_generator.step(batch_size=2000)["mask"]
        D = D.mean(0, keepdim=True)
        self.D = 1 / D.sqrt()
        
    def __call__(self, x: Tensor):
        return self.kspace_to_im(self.im_to_kspace(x) / self.D)

    def i(self, x: Tensor):
        return self.kspace_to_im(self.im_to_kspace(x) * self.D)
    
    def FhD(self, y: Tensor):
        return self.kspace_to_im(y * self.D)

class ENSURELoss(SureGaussianLoss):
    def __init__(self, sigma, physics_generator, tau=0.01, rng=None):
        super().__init__(sigma=sigma, tau=tau, rng=rng)
        w = W(physics_generator)
        self.metric = lambda y: w.FhD(y)

    def div(self, x_net, y, f, physics):
        b = torch.empty_like(y).normal_(generator=self.rng)
        x2 = f(physics.A(physics.A_adjoint(y) + b * self.tau), physics)
        return (b * (x2 - x_net) / self.tau).reshape(y.size(0), -1).mean(1)

    def forward(self, y, x_net, physics, model, **kwargs):
        y1 = physics.A(x_net)
        div = 2 * self.sigma2 * self.div(x_net, y, model, physics)
        mse = self.metric(y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        loss_sure = mse + div - self.sigma2

        return loss_sure