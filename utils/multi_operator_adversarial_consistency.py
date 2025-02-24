from __future__ import annotations
from typing import TYPE_CHECKING
import torch.nn as nn
from torch import Tensor
import torch

from deepinv.physics.mri import MRI
from .consistency import (
    UnsupAdversarialDiscriminatorLoss,
    UnsupAdversarialGeneratorLoss,
)
from .uair import MultiOperatorMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics

def physics_like(physics: Physics, y: Tensor) -> Physics:
    """Return physics with params based on inputs.

    :param deepinv.physics.Physics physics: input physics.
    :param torch.Tensor y: input tensor.
    :return deepinv.physics.Physics: new physics.
    """
    if isinstance(physics, MRI):
        return MRI(img_size=y.shape, device=y.device)
    else:
        return physics.__class__(tensor_size=y.shape, device=y.device)


class MultiOperatorUnsupAdversarialGeneratorLoss(
    MultiOperatorMixin, UnsupAdversarialGeneratorLoss
):
    """Multi-operator unsupervised adversarial loss for generator.

    Extends unsupervised adversarial loss by sampling new physics and new data every iteration.

    Proposed in `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param callable dataloader_factory: callable that returns a dataloader that returns new samples
    """

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        D: nn.Module = None,
        epoch=None,
        **kwargs,
    ):
        self.reset_iter(epoch=epoch)

        y_tilde = self.next_data()[1].to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)

        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)

        physics_full = physics_like(physics, y_hat)
        x_tilde = physics_full.A_adjoint(y_tilde)
        x_hat = physics_full.A_adjoint(y_hat)

        return self.adversarial_loss(x_tilde, x_hat, D)


class MultiOperatorUnsupAdversarialDiscriminatorLoss(
    MultiOperatorMixin, UnsupAdversarialDiscriminatorLoss
):
    """Multi-operator unsupervised adversarial loss for discriminator.

    Extends unsupervised adversarial loss by sampling new physics and new data every iteration.

    Proposed in `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.

    :param float weight_adv: weight for adversarial loss, defaults to 1.0
    :param torch.nn.Module D: discriminator network. If not specified, D must be provided in forward(), defaults to None.
    :param str device: torch device, defaults to "cpu"
    :param callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param callable dataloader_factory: callable that returns a dataloader that returns new samples
    """

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics: Physics,
        D: nn.Module = None,
        epoch=None,
        **kwargs,
    ):
        self.reset_iter(epoch=epoch)

        y_tilde = self.next_data()[1].to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)

        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)

        physics_full = physics_like(physics, y_hat)
        x_tilde = physics_full.A_adjoint(y_tilde)
        x_hat = physics_full.A_adjoint(y_hat)

        return self.adversarial_loss(x_tilde, x_hat, D)