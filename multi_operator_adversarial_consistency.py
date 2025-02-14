from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy
import torch.nn as nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader

from consistency import UnsupAdversarialDiscriminatorLoss, UnsupAdversarialGeneratorLoss
if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.physics.forward import Physics

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
        y_tilde = self.next_data()
        physics_new = self.next_physics(physics)
        y_hat = physics_new.A(x_net)
        
        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)
        
        x_tilde = physics.A_adjoint(y_tilde)
        x_hat   = physics.A_adjoint(y_hat)

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

class ColeDiscriminator(nn.Module):
    def __init__(self, h=320, d_dim=128, d_blocks=4, in_channels=2):
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
        self.final = nn.Linear(d_dim * h * h, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv(x)
        
        for i in range(0, len(self.blocks), 2):
            x = self.blocks[i](x)
            x += self.blocks[i+1](x)

        return self.sigmoid(self.final(self.flatten(x)))