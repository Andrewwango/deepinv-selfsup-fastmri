from __future__ import annotations

from typing import Union, Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.metric.metric import Metric
from deepinv.transform.base import Transform, TransformParam
from deepinv.physics.mri import MRI, MRIMixin
from deepinv.physics.noise import GaussianNoise, NoiseModel
from deepinv.transform import Transform, Rotate, Shift


class RandomNoise(Transform):
    """Random noise transform.

    For now, only Gaussian noise is supported. Override this class and replace the `sigma` parameter for other noise models.

    This transform is reproducible: for given param dict `noise_model`, the transform is deterministic.

    Note the inverse transform is not well-defined for this transform.

    :param str noise_type: noise distribution, currently only supports Gaussian noise.
    :param int, tuple[int, int] sigma: noise parameter or range to pick randomly.
    """

    def __init__(
        self,
        *args,
        noise_type: str = "gaussian",
        sigma: Union[int, Tuple[int, int]] = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        if noise_type == "gaussian":
            self.noise_class = GaussianNoise
        else:
            raise ValueError(f"Noise type {noise_type} not supported.")

    def _get_params(self, *args) -> dict:
        if isinstance(sr := self.sigma, tuple):
            sigma = (
                torch.rand(self.n_trans, generator=self.rng, device=self.rng.device) * (sr[1] - sr[0])
            ) + sr[0]
        else:
            sigma = [self.sigma] * self.n_trans
        # TODO reproducible, different rng when self.n_trans > 1
        return {
            "noise_model": [
                self.noise_class(sigma=s, rng=self.rng if i == 0 else None)
                for i, s in enumerate(sigma)
            ]
        }

    def _transform(
        self, y: Tensor, noise_model: Iterable[NoiseModel] = [], **kwargs
    ) -> Tensor:
        mask = (y != 0).int()
        return torch.cat([n(y) * mask for n in noise_model])

    def inverse(self, *args, **kwargs):
        raise ValueError("Noise transform is not invertible.")


class RandomPhaseError(Transform):
    """Random phase error transform.

    This transform adds a phase error to k-space using:

    :math:`Ty=\exp(-i\phi_k)y` where :math:`\phi_k=\pi\alpha s_e` if :math:`k` is an even index,
    or :math:`\phi_k=\pi\alpha s_o` if odd, and where :math:`\alpha` is a scale parameter,
    and :math:`s_o,s_e\sim U(-1,1)`.

    This transform is reproducible: for given param dict `se, so`, the transform is deterministic.

    :param int, tuple[int, int] scale: scale parameter or range to pick randomly.
    """

    def __init__(self, *args, scale: Union[int, Tuple[int, int]] = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale

    def _get_params(self, *args) -> dict:
        if isinstance(s := self.scale, tuple):
            scale = (
                torch.rand((1, self.n_trans), generator=self.rng, device=self.rng.device) * (s[1] - s[0])
            ) + s[0]
        else:
            scale = self.scale

        se, so = (
            2
            * torch.pi
            * scale
            * torch.rand((2, self.n_trans), generator=self.rng, device=self.rng.device)
            - torch.pi * scale
        )
        return {"se": se, "so": so}

    def _transform(
        self,
        y,
        se: Union[torch.Tensor, Iterable, TransformParam] = [],
        so: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> Tensor:
        out = []
        for _se, _so in zip(se, so):
            shift = MRIMixin.to_torch_complex(torch.zeros_like(y))
            shift[..., 0::2] = torch.exp(-1j * _se)  # assume readouts in w
            shift[..., 1::2] = torch.exp(-1j * _so)
            out += [y * MRIMixin.from_torch_complex(shift)]
        return torch.cat(out)


class VORTEXLoss(Loss):
    r"""Measurement data augmentation loss.

    Performs data augmentation in measurement domain as proposed by
    `VORTEX: Physics-Driven Data Augmentations Using Consistency Training for Robust Accelerated MRI Reconstruction <https://arxiv.org/abs/2111.02549>`_.

    The loss is defined as follows:

    :math:`\mathcal{L}(T_e\inverse{y,A},\inverse{T_i y,A T_e^{-1}})`

    where :math:`T_1` is a random :class:`deepinv.transform.Transform` defined in k-space and
    :math:`T_2` is a random :class:`deepinv.transform.Transform` defined in image space.
    By default, for :math:`T_1` we add random noise :class:`deepinv.transform.RandomNoise` and random phase error :class:`deepinv.transform.RandomPhaseError`.
    By default, for :math:`T_2` we use random shift :class:`deepinv.transform.Shift` and random rotates :class:`deepinv.transform.Rotate`.

    .. note::

        See :ref:`transform` for a guide on all available transforms, and how to compose them. For example, you can easily
        compose further transforms such as  ``Rotate(rng=rng, multiples=90) | Scale(factors=[0.75, 1.25], rng=rng) | Reflect(rng=rng)``.

    .. note::

        For now, this loss is only available for MRI problems, but it is easily generalisable to other problems.

    :param deepinv.transform.Transform T_1: k-space transform.
    :param deepinv.transform.Transform T_2: image transform.
    :param deepinv.loss.metric.Metric, torch.nn.Module metric: metric for calculating loss.
    :param bool no_grad: if ``True``, only propagate gradients through augmented branch as per original paper,
        if ``False``, propagate through both branches.
    :param torch.Generator rng: torch random number generator to pass to transforms.
    """

    def __init__(
        self,
        T_1: Transform = None,
        T_2: Transform = None,
        metric: Union[Metric, nn.Module] = torch.nn.MSELoss(),
        no_grad: bool = True,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.T_1 = (
            T_1
            if T_1 is not None
            else RandomPhaseError(scale=0.1, rng=rng) * RandomNoise(rng=rng)
        )
        self.T_2 = (
            T_2
            if T_2 is not None
            else Shift(shift_max=0.1, rng=rng) | Rotate(rng=rng, limits=15)
        )
        self.no_grad = no_grad

    class TransformedMRI(MRI):
        """Pre-multiply physics with transform.

        :param deepinv.physics.MRI: original MRI physics
        :param deepinv.transform.Transform: transform object
        :param dict transform_params: fixed parameters for deterministic transform.
        """

        def __init__(
            self,
            physics: MRI,
            transform: Transform,
            transform_params: dict,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                img_size=physics.img_size,
                mask=physics.mask,
                device=physics.device,
                three_d=physics.three_d,
                **kwargs,
            )
            self.transform = transform
            self.transform_params = transform_params

        def A(self, x, *args, **kwargs):
            return super().A(
                self.transform.inverse(x, **self.transform_params), *args, **kwargs
            )

        def A_adjoint(self, y, *args, **kwargs):
            return self.transform(
                super().A_adjoint(y, *args, **kwargs), **self.transform_params
            )

    def forward(self, x_net: Tensor, y: Tensor, physics: MRI, model, **kwargs):
        r"""
        VORTEX loss forward pass.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return: (:class:`torch.Tensor`) loss, the tensor size might be (1,) or (batch size,).
        """
        if self.no_grad:
            x_net = x_net.detach()

        # Sample image transform
        e_params = self.T_2.get_params(x_net)

        # Augment input
        x_aug = self.T_2(physics.A_adjoint(self.T_1(y)), **e_params)

        # Pass through network
        physics2 = self.TransformedMRI(physics, self.T_2, e_params)
        
        if isinstance(model, SplittingLoss.SplittingModel):
            _model = model.model
            print("Splitting base model in Noise2Recon")
        else:
            _model = model
        
        x_aug_net = _model(physics2(x_aug), physics2)

        return self.metric(self.T_2(x_net, **e_params), x_aug_net)
