import torch.nn as nn
import torch


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: list[int, int], stride: int = 1):
        """
        Initialize a block containing Batch Normalization, Leaky ReLU, and Convolution layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size  (list[int, int]): Size of the convolutional kernel.
            stride (int, optional): Subsampling factor/stride. Defaults to 1.
        """
        super(ConvolutionBlock, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode="zeros"
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: list[int, int], stride: int = 1):
        """
        A block containing Batch Normalization, Leaky ReLU, Convolution, and Upsample layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size  (list[int, int]): Size of the convolutional kernel.
            stride (int, optional): Stride for the convolutional operation. Defaults to 1.
            subsample (int, optional): Subsampling factor/stride for convolution. Defaults to 1.
        """
        super(UpsampleConvBlock, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Residual shortcut connection block for neural networks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Subsampling factor/stride. Defaults to 1.
        """
        super(ResidualShortcut, self).__init__()
        self.should_process = False
        self.processing_block = None

        # Check if shortcut processing is needed
        if in_channels != out_channels or stride != 1:
            self.should_process = True
            self.processing_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.should_process:
            x = self.processing_block(x)
        return x + y


class UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Upsample shortcut connection block for neural networks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Subsampling factor/stride.
        """
        super(UpsampleShortcut, self).__init__()
        self.should_process = False
        self.processing_block = None

        # Check if shortcut processing is needed
        if in_channels != out_channels:
            self.should_process = True
            self.processing_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding_mode="zeros"),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.should_process:
            x = self.processing_block(x)
        return x + y


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_stride: int = 1):
        """
        Basic building block for residual networks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            init_stride (int, optional): Initial subsampling factor/stride. Defaults to 1.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = ConvolutionBlock(in_channels, out_channels, [3, 3], stride=init_stride)
        self.residual = ConvolutionBlock(out_channels, out_channels, [3, 3])
        self.shortcut = ResidualShortcut(in_channels, out_channels, stride=init_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)


class UpsampleBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_stride: int = 1):
        """
        Upsample basic building block for residual networks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            init_subsample (int, optional): Initial subsampling factor/stride. Defaults to 1.
        """
        super(UpsampleBasicBlock, self).__init__()
        self.conv1 = UpsampleConvBlock(in_channels, out_channels, [3, 3], stride=init_stride)
        self.residual = ConvolutionBlock(out_channels, out_channels, [3, 3])
        self.shortcut = UpsampleShortcut(in_channels, out_channels, stride=init_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, is_first_layer: bool = False):
        """
        Residual block for building neural network architectures.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of repetitions for BasicBlock within the ResidualBlock.
            is_first_layer (bool, optional): Flag indicating whether this is the first layer. Defaults to False.
        """
        super(ResidualBlock, self).__init__()
        layers = []

        block = BasicBlock(in_channels, out_channels, init_stride=1)
        layers.append(block)
        for i in range(1, num_blocks):
            init_stride = 1 if i != num_blocks - 1 or is_first_layer else 2
            block = BasicBlock(out_channels, out_channels, init_stride)
            layers.append(block)

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UpsamplingResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        """
        Upsampling residual block for building neural network architectures.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of repetitions for BasicBlock within the UpsamplingResidualBlock.
        """
        super(UpsamplingResidualBlock, self).__init__()

        self.blocks = nn.Sequential(
            *[
                UpsampleBasicBlock(in_channels, out_channels) if i == 0 else BasicBlock(out_channels, out_channels)
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
