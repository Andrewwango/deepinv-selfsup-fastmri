from __future__ import annotations

from typing import Union, Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform, TransformParam
from deepinv.physics.mri import MRI, MRIMixin
from deepinv.physics.noise import GaussianNoise, NoiseModel

class NoiseTransform(Transform):
    def __init__(self, *args, sigma: Union[int, Tuple[int, int]] = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
    
    def _get_params(self, *args) -> dict:
        if isinstance(sr := self.sigma, tuple):
            sigma = (torch.rand(self.n_trans, generator=self.rng) * (sr[1] - sr[0])) + sr[0]
        else:
            sigma = [self.sigma] * self.n_trans
        # TODO reproducible, different rng when self.n_trans > 1
        return {"noise_model": [GaussianNoise(sigma=s, rng=self.rng if i == 0 else None) for i, s in enumerate(sigma)]}
    
    def _transform(self, y: Tensor, noise_model: Iterable[NoiseModel] = [], **kwargs) -> Tensor:
        mask = (y != 0).int()
        return torch.cat([n(y) * mask for n in noise_model])
    
    def inverse(self, *args, **kwargs):
        raise ValueError("Noise transform is not invertible.")

class RandomPhaseShift(Transform):
    def __init__(self, *args, scale: Union[int, Tuple[int, int]] = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def _get_params(self, *args) -> dict:
        if isinstance(s := self.scale, tuple):
            scale = (torch.rand((1, self.n_trans), generator=self.rng) * (s[1] - s[0])) + s[0]
        else:
            scale = self.scale
        
        se, so = 2 * torch.pi * scale * torch.rand((2, self.n_trans), generator=self.rng, device=self.rng.device) - torch.pi * scale
        return {"se": se, "so": so}

    def _transform(self, y, se: Union[torch.Tensor, Iterable, TransformParam] = [], so: Union[torch.Tensor, Iterable, TransformParam] = [], **kwargs) -> Tensor:
        out = []
        for (_se, _so) in zip(se, so):
            shift = MRIMixin.to_torch_complex(torch.zeros_like(y))
            shift[..., 0::2] = torch.exp(-1j * _se) # assume readouts in w
            shift[..., 1::2] = torch.exp(-1j * _so)            
            out += [y * MRIMixin.from_torch_complex(shift)]
        return torch.cat(out)

from deepinv.transform import Transform, Rotate, Shift, Scale, Reflect

class VORTEXLoss(Loss):
    def __init__(
        self,
        T_e: Transform = None,
        T_i: Transform = None,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        no_grad: bool = True,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.T_e = T_e if T_e is not None else Shift(shift_max=0.1, rng=rng) | Rotate(rng=rng, limits=15) #| Rotate(rng=rng, multiples=90) | Scale(factors=[0.75, 1.25], rng=rng) | Reflect(rng=rng)
        self.T_i = T_i if T_i is not None else RandomPhaseShift(scale=0.1, rng=rng) * NoiseTransform(rng=rng)
        self.no_grad = no_grad

    class TransformedMRI(MRI):
        def __init__(self, physics: MRI, transform: Transform, transform_params: dict, *args, **kwargs):
            super().__init__(*args, img_size=physics.img_size, mask=physics.mask, device=physics.device, three_d=physics.three_d, **kwargs)
            self.transform = transform
            self.transform_params = transform_params
        
        def A(self, x, mask=None, **kwargs):
            return super().A(self.transform.inverse(x, **self.transform_params), mask, **kwargs)
        
        def A_adjoint(self, y, mask = None, **kwargs):
            return self.transform(super().A_adjoint(y, mask, **kwargs), **self.transform_params)

    def forward(self, x_net: Tensor, y: Tensor, physics: MRI, model, **kwargs):
        if self.no_grad:
            # Only propagate gradients through augmented branch
            x_net = x_net.detach()
        
        # Sample E transform
        e_params = self.T_e.get_params(x_net)
        
        # Augment input
        x_aug = self.T_e(physics.A_adjoint(self.T_i(y)), **e_params)

        # Pass through network
        physics2 = self.TransformedMRI(physics, self.T_e, e_params)
        x_aug_net = model(physics2(x_aug), physics2)

        return self.metric(self.T_e(x_net, **e_params), x_aug_net)