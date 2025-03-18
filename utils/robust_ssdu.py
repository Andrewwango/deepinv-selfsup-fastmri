
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
        denoi_loss = mask * (physics.A(x_net) - y) * (1 + 1/self.a2)

        return recon_loss + denoi_loss

    def adapt_model(self, model: torch.nn.Module) -> RobustSplittingModel:
        if isinstance(model, self.RobustSplittingModel):
            return model
        else:
            return self.RobustSplittingModel(model, mask_generator=self.mask_generator, noise_model=self.noise_model)

    class RobustSplittingModel(SplittingLoss.SplittingModel):
        def __init__(self, model, mask_generator, noise_model):
            super().__init__()
            self.model = model
            self.eval_n_samples = 1
            self.mask = 0
            self.mask_generator = mask_generator
            self.noise_model = noise_model
            self.eval_split_input = False
            self.eval_split_output = False

        def split(self, mask, y, physics=None):
            print("Noisy splitting")
            y1, physics1 = SplittingLoss.split(mask, y, physics)
            return mask * self.noise_model(y1), physics1