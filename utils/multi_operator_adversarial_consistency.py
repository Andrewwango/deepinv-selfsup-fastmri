from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy
import torch.nn as nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader

from .consistency import UnsupAdversarialDiscriminatorLoss, UnsupAdversarialGeneratorLoss

if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.physics.forward import Physics

from deepinv.physics.mri import MRI

class MultiOperatorMixin:
    def __init__(
        self,
        physics_generator_factory: Callable[..., PhysicsGenerator],
        dataloader_factory: Callable[..., DataLoader],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.physics_generator = physics_generator_factory()
        self.iterator = iter(dataloader_factory())

    def next_physics(self, physics: Physics):
        physics_cur = deepcopy(physics)
        params = self.physics_generator.step()
        physics_cur.update_parameters(**params)
        return physics_cur
    
    def next_data(self):
        return next(self.iterator)[1]

class MultiOperatorUnsupAdversarialGeneratorLoss(MultiOperatorMixin, UnsupAdversarialGeneratorLoss):
    def forward(self, y: Tensor, x_net: Tensor, physics: Physics, D: nn.Module = None, **kwargs):
        y_tilde = self.next_data().to(x_net.device)
        physics_new = self.next_physics(physics)
        y_hat = physics_new.A(x_net)
        
        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)
        
        physics_full = MRI(img_size=y_hat.shape, device=y_hat.device)
        x_tilde = physics_full.A_adjoint(y_tilde)
        x_hat   = physics_full.A_adjoint(y_hat)

        return self.adversarial_loss(x_tilde, x_hat, D)

class MultiOperatorUnsupAdversarialDiscriminatorLoss(MultiOperatorMixin, UnsupAdversarialDiscriminatorLoss):
    def forward(self, y: Tensor, x_net: Tensor, physics: Physics, D: nn.Module = None, **kwargs):
        y_tilde = self.next_data()
        physics_new = self.next_physics(physics)
        y_hat = physics_new.A(x_net)
        
        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)
        
        x_tilde = physics.A_adjoint(y_tilde)
        x_hat   = physics.A_adjoint(y_hat)

        return self.adversarial_loss(x_tilde, x_hat, D)

from typing import Tuple
from math import prod
class SkipConvDiscriminator(nn.Module):
    """Simple residual convolution discriminator architecture.

    Consists of convolutional blocks with skip connections with a final dense layer followed by sigmoid.

    :param tuple img_size: tuple of ints of input image size
    :param int d_dim: hidden dimension
    :param int d_blocks: number of conv blocks
    :param int in_channels: number of input channels
    """
    def __init__(self, img_size: Tuple[int, int] = (320, 320), d_dim: int = 128, d_blocks: int = 4, in_channels: int = 2):
        super().__init__()
        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU()
            )

        self.initial_conv = conv_block(in_channels, d_dim)
        
        self.blocks = nn.ModuleList()
        for _ in range(d_blocks):
            self.blocks.append(conv_block(d_dim, d_dim))
            self.blocks.append(conv_block(d_dim, d_dim))
        
        self.flatten = nn.Flatten()
        self.final = nn.Linear(d_dim * prod(img_size), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)
        
        for i in range(0, len(self.blocks), 2):
            x1 = self.blocks[i](x)
            x2 = x1 + self.blocks[i+1](x)
            x = x2

        return self.sigmoid(self.final(self.flatten(x))).squeeze()