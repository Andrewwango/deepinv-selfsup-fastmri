from typing import Tuple
from math import prod
from torch import nn, Tensor


class SkipConvDiscriminator(nn.Module):
    """Simple residual convolution discriminator architecture.

    Architecture taken from `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.

    Consists of convolutional blocks with skip connections with a final dense layer followed by sigmoid.

    :param tuple img_size: tuple of ints of input image size
    :param int d_dim: hidden dimension
    :param int d_blocks: number of conv blocks
    :param int in_channels: number of input channels
    :param bool use_sigmoid: use sigmoid activation at output.
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
