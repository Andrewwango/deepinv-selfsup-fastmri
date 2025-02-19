from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from torch.nn import MSELoss
from deepinv.loss.loss import Loss
from deepinv.models.base import Reconstructor

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

class NoTransform(Transform):
    def _get_params(self, *args):
        return {}
    
    def _transform(self, x, **params):
        return x

from deepinv.transform.base import Transform
class TransformedMRI(MRI):
    def __init__(self, transform: Transform, transform_params: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform
        self.transform_params = transform_params
    
    def A(self, x, mask=None, **kwargs):
        return super().A(self.transform.inverse(x, **self.transform_params), mask, **kwargs)
    
    def A_adjoint(self, y, mask = None, **kwargs):
        return self.transform(super().A_adjoint(y, mask, **kwargs), **self.transform_params)

class VORTEXLoss(Loss):
    def __init__(
        self,
        transform_equiv: Transform,
        transform_invar: Transform,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.T_e = transform_equiv
        self.T_i = transform_invar

    def forward(self, x_net, y, physics: MRI, model, **kwargs):
        e_params = self.T_e.get_params(x_net)

        x1 = self.T_e(x_net, **e_params)
        
        yi = self.T_i(y)
        xi = physics.A_adjoint(yi)
        xe = self.T_e(xi, **e_params)
        #physics_full = MRI(img_size=y.shape, device=physics.device)
        #ye = physics_full(xe)
        #x2 = model(ye, physics_full)
        #ye = physics.A(xe)
        #x2 = model(ye, physics)
        physics2 = TransformedMRI(self.T_e, e_params, img_size=y.shape, mask=physics.mask, device=physics.device)
        ye = physics2(xe)
        x2 = model(ye, physics2)

        return self.metric(x1, x2)
