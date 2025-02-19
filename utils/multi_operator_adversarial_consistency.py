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
        dataloader_factory: Callable[..., DataLoader] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.physics_generator = physics_generator_factory()
        if dataloader_factory is not None:
            self.dataloader = dataloader_factory()
            self.prev_epoch = -1
            self.reset_iter(epoch=0)

    def next_physics(self, physics: Physics, batch_size=1):
        physics_cur = deepcopy(physics)
        params = self.physics_generator.step(batch_size=batch_size)
        physics_cur.update_parameters(**params)
        return physics_cur
    
    def next_data(self):
        return next(self.iterator)[1]
    
    def reset_iter(self, epoch):
        if epoch == self.prev_epoch:
            pass
        elif epoch == self.prev_epoch + 1:
            self.iterator = iter(self.dataloader)
            self.prev_epoch += 1
        else:
            raise ValueError("This shouldn't happen...")

    def physics_like(self, y):
        return MRI(img_size=y.shape, device=y.device)

class MultiOperatorUnsupAdversarialGeneratorLoss(MultiOperatorMixin, UnsupAdversarialGeneratorLoss):
    def forward(self, y: Tensor, x_net: Tensor, physics: Physics, D: nn.Module = None, epoch=None, **kwargs):
        self.reset_iter(epoch=epoch)

        y_tilde = self.next_data().to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)
        
        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)
        
        physics_full = self.physics_like(y_hat)
        x_tilde = physics_full.A_adjoint(y_tilde)
        x_hat   = physics_full.A_adjoint(y_hat)

        return self.adversarial_loss(x_tilde, x_hat, D)

class MultiOperatorUnsupAdversarialDiscriminatorLoss(MultiOperatorMixin, UnsupAdversarialDiscriminatorLoss):
    def forward(self, y: Tensor, x_net: Tensor, physics: Physics, D: nn.Module = None, epoch=None, **kwargs):
        self.reset_iter(epoch=epoch)
        
        y_tilde = self.next_data().to(x_net.device)
        physics_new = self.next_physics(physics, batch_size=len(x_net))
        y_hat = physics_new.A(x_net)
        
        assert y_tilde.shape == y_hat.shape
        assert not torch.all(physics.mask == physics_new.mask)
        
        physics_full = self.physics_like(y_hat)
        x_tilde = physics_full.A_adjoint(y_tilde)
        x_hat   = physics_full.A_adjoint(y_hat)

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

    def __init__(
        self,
        img_size: Tuple[int, int] = (320, 320),
        d_dim: int = 128,
        d_blocks: int = 4,
        in_channels: int = 2,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(),
            )

        self.initial_conv = conv_block(in_channels, d_dim)

        self.blocks = nn.ModuleList()
        for _ in range(d_blocks):
            self.blocks.append(conv_block(d_dim, d_dim))
            self.blocks.append(conv_block(d_dim, d_dim))

        self.flatten = nn.Flatten()
        self.final = nn.Linear(d_dim * prod(img_size), 1)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_conv(x)

        for i in range(0, len(self.blocks), 2):
            x1 = self.blocks[i](x)
            x2 = x1 + self.blocks[i + 1](x)
            x = x2
        
        y = self.final(self.flatten(x))
        return self.sigmoid(y).squeeze() if self.use_sigmoid else y.squeeze()



from typing import List
class SkipConvDiscriminator2(nn.Module):
    """Simple residual convolution discriminator architecture.

    Consists of convolutional blocks with skip connections with a final dense layer followed by sigmoid.

    :param list[int] hidden_dims: hidden dimensions
    :param int d_blocks: number of conv blocks
    :param int in_channels: number of input channels
    :param int h: image size
    :param bool use_sigmoid: whether to use sigmoid
    """
    def __init__(
        self, 
        in_channels: int = 2,
        hidden_dims: List[int] = [64, 128, 256, 512, 1],
        h: int = 320,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        
        kern, stride, padding = 4, 2, 1
        layers = []
        dims = [in_channels] + hidden_dims
        
        layers.extend([
            *[
                layer
                for i in range(len(dims) - 1)
                for layer in (
                    nn.Conv2d(dims[i], dims[i+1], kern, stride, padding),
                    nn.BatchNorm2d(dims[i+1]) if i < len(dims)-2 else None,
                    nn.LeakyReLU(0.2, inplace=True) if i < len(dims)-2 else None
                )
                if layer is not None
            ]
        ])
        
        self.blocks = nn.Sequential(*layers)
        
        for _ in range(len(hidden_dims)):
            h = (h + 2*padding - kern) // stride + 1
            
        self.flatten = nn.Flatten()
        self.final = nn.Linear(h*h, 1)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        y = self.final(self.flatten(x))
        return self.sigmoid(y).squeeze() if self.use_sigmoid else y.squeeze()