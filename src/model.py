import torch.nn as nn
from src.building_blocks import ConvolutionBlock, ResidualBlock, UpsamplingResidualBlock, ResidualShortcut


class MangaLineExtractor(nn.Module):
    def __init__(self):
        super(MangaLineExtractor, self).__init__()

        # Encoder Blocks
        encoder_configs = [
            (1, 24, 2, True),
            (24, 48, 3, False),
            (48, 96, 5, False),
            (96, 192, 7, False),
            (192, 384, 12, False),
        ]

        self.encoder_blocks = nn.ModuleList(
            [
                ResidualBlock(in_channels, out_channels, num_blocks, is_first_layer)
                for in_channels, out_channels, num_blocks, is_first_layer in encoder_configs
            ]
        )

        # Decoder Blocks with Shortcut Connections
        decoder_configs = [(384, 192, 7), (192, 96, 5), (96, 48, 3), (48, 24, 2)]
        self.decoder_blocks = nn.ModuleList(
            [
                UpsamplingResidualBlock(in_channels, out_channels, num_blocks)
                for in_channels, out_channels, num_blocks in decoder_configs
            ]
        )

        self.shortcuts = nn.ModuleList(
            [
                ResidualShortcut(in_channels, out_channels)
                for in_channels, out_channels in [(192, 192), (96, 96), (48, 48), (24, 24)]
            ]
        )

        # Final Block
        self.final_block = ResidualBlock(24, 16, 2, True)
        self.final_conv = ConvolutionBlock(16, 1, [1, 1], stride=1)

    def forward(self, x):
        encoder_outputs = [x]
        for encoder_block in self.encoder_blocks:
            encoder_outputs.append(encoder_block(encoder_outputs[-1]))

        decoder_outputs = [self.decoder_blocks[0](encoder_outputs[-1])]
        for i, (decoder_block, shortcut) in enumerate(zip(self.decoder_blocks[1:], self.shortcuts), 1):
            decoder_outputs.append(decoder_block(shortcut(encoder_outputs[-i - 1], decoder_outputs[-1])))

        y = self.final_conv(self.final_block(decoder_outputs[-1]))

        return y
