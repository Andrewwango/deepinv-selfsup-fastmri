
from __future__ import annotations
import torch
import deepinv as dinv
from deepinv.loss.measplit import WeightedSplittingLoss, SplittingLoss
from deepinv.physics import GaussianNoise, NoiseModel
from deepinv.physics.generator import (
    BaseMaskGenerator,
    BernoulliSplittingMaskGenerator,
)

class RobustSplittingLoss(WeightedSplittingLoss):
    def __init__(
        self,
        mask_generator: BernoulliSplittingMaskGenerator,
        physics_generator: BaseMaskGenerator,
        noise_model: GaussianNoise = GaussianNoise(sigma=0.1),
        alpha: float = 0.75,
        eps: float = 1e-9,
    ):
        super().__init__(mask_generator, physics_generator, eps=eps)
        self.a2 = alpha ** 2
        self.noise_model = noise_model
        self.noise_model.update_parameters(sigma=noise_model.sigma * self.a2)
        
    def forward(self, x_net, y, physics, model, **kwargs):
        recon_loss = super().forward(x_net, y, physics, model, **kwargs)
        
        mask = model.get_mask() * getattr(physics, "mask", 1.0) # M_\lambda\cap\omega 
        residual = mask * (physics.A(x_net) - y) * (1 + 1/self.a2)

        return recon_loss + (residual**2).mean()

    def adapt_model(self, model: torch.nn.Module) -> RobustSplittingModel:
        if isinstance(model, self.RobustSplittingModel):
            return model
        else:
            return self.RobustSplittingModel(model, mask_generator=self.mask_generator, noise_model=self.noise_model)

    class RobustSplittingModel(SplittingLoss.SplittingModel):
        def __init__(self, model, mask_generator, noise_model):
            super().__init__(model, split_ratio=None, mask_generator=mask_generator, eval_n_samples=1, eval_split_input=False, eval_split_output=False, pixelwise=True)
            self.noise_model = noise_model

        def split(self, mask, y, physics=None):
            y1, physics1 = SplittingLoss.split(mask, y, physics)
            return mask * self.noise_model(y1), physics1