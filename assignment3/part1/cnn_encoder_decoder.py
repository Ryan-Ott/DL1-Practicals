################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # * Note: output size = floor(((input_size - kernel_size + 2*padding) / stride) + 1)
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2),  # 28x28 -> 14x14
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.GELU(),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2),  # 14x14 -> 7x7
            nn.GELU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.GELU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2),  # 7x7 -> 4x4
            nn.GELU(),
            nn.Flatten(),  # Turns the 2*num_filters output of the 4x4 conv into a 1D vector
        )
        self.mean_layer = nn.Linear(2*4*4*num_filters, z_dim)
        self.log_std_layer = nn.Linear(2*4*4*num_filters, z_dim)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / 15 * 2.0 - 1.0  # Move images between -1 and 1
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.net(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 4*4*2*num_filters),
            nn.GELU()
        )
        # * Note: output size = stride * (input_size - 1) + kernel_size - 2 * padding + output_padding
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2, output_padding=0),  # 4x4 -> 7x7 thanks to adjusted output_padding
            nn.GELU(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),  # 7x7 -> 7x7
            nn.GELU(),
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, padding=1, stride=2, output_padding=1),  # 7x7 -> 14x14
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.GELU(),
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, padding=1, stride=2, output_padding=1),  # 14x14 -> 28x28
        )
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a softmax applied on it.
                Shape: [B,num_input_channels,28,28]
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        #######################
        # END OF YOUR CODE    #
        #######################
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
