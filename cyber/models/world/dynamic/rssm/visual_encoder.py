"""
Aspect of the code inspired/reference from:
google-research/planet: https://github.com/google-research/planet
planet-torch: https://github.com/abhayraw1/planet-torch
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 # this is the convention used in PyTorch

import math


class VisualEncoder(nn.Module):
    """
    A simple convolutional encoder for visual observations.
    """

    def __init__(self, embedding_size=1024, image_side=64, num_layers=4, activation_function="relu", **kwargs):
        """
        Args:
            embedding_size (int): The size of the output embedding. Recommended to be a power of 2.
            image_side (int): The side length of the square input image. Must be a power of 2. Assumes 3 color channels.
            num_layers (int): Number of convolution layers in the encoder
            activation_function (str): The activation function to use.
        """
        super().__init__()
        self.starting_side = image_side
        power = int(math.log2(image_side))
        self.input_size = (3, image_side, image_side)
        assert int(2**power) == image_side, "image_side must be a power of 2"
        assert power >= num_layers, "num_layers must be less than or equal to the log base 2 of image_side"
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        for i in range(num_layers):
            out_channels = 2 ** (i + 5)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            in_channels = out_channels
        feats = int(image_side * image_side * 16 / (2**num_layers))  # 32 is the starting number of channels, 2 is the stride
        self.fc = nn.Linear(feats, embedding_size)

    def forward(self, observation):
        assert observation.size()[-3:] == self.input_size, "Input observation size does not match the expected size"
        encoded = observation
        for layer in self.conv_layers:
            encoded = self.act_fn(layer(encoded))
        encoded = torch.flatten(encoded, start_dim=1)  # dims: (B, C, H, W) -> (B, C*H*W)
        encoded = self.fc(encoded)
        return encoded


class VisualDecoder(nn.Module):
    """
    A simple deconvolutional decoder for visual observations.
    """

    def __init__(self, input_size: int = 1024, image_size: int = 64, num_layers: int = 4, activation_function="relu", **kwargs):
        """
        Args:
            input_size (int): The size of the input.
            output_shape (int): The side length of the square output image. Must be a power of 2. Assumes 3 color channels.
            num_layers (int): Number of convolution layers in the encoder
        """
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.input_size = input_size
        self.output_shape = image_size
        power = int(math.log2(image_size))
        assert int(2**power) == image_size, "output_shape must be a power of 2"
        assert num_layers <= power, "num_layers must be less than or equal to the log base 2 of output_shape"
        in_channels = 32 * (2 ** (num_layers - 1))
        self.starting_feature_size = in_channels
        self.starting_image_side = 2 ** (power - num_layers + 1)
        self.fc = nn.Linear(input_size, int(self.starting_image_side * self.starting_image_side * in_channels))
        self.conv_layers = nn.ModuleList()
        for _i in range(num_layers - 1):
            out_channels = in_channels // 2
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            self.conv_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            in_channels = out_channels
        self.conv_layers.append(nn.Conv2d(32, 3, 3, stride=1, padding=1))

    def forward(self, state):
        hidden = self.fc(state)
        hidden = hidden.view(-1, self.starting_feature_size, self.starting_image_side, self.starting_image_side)
        for layer in self.conv_layers:
            hidden = self.act_fn(layer(hidden))
        return hidden
