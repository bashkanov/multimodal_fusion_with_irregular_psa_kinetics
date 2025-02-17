# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Originally from: https://github.com/NVIDIA/DeepLearningExamples/blob/ff6f7c6532e50f0d6b7268b3e3ee66083dd260c7/PyTorch/Segmentation/nnUNet/models/unet.py
# Implementation of DAFT modules highly influenced by https://github.com/ai-med/DAFT
# Modified by  Oleksii Bashkanov/OVGU. All rights reserved. (c) 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}

dropouts = {
    "Dropout": nn.Dropout,
    "Dropout3d": nn.Dropout3d,
    "Dropout2d": nn.Dropout2d,
}


def get_norm(name, out_channels):
    if "groupnorm" in name:
        return nn.GroupNorm(32, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim, groups=1, bias=False):
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, groups=in_channels if groups == -1 else groups,
                bias=bias)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)


def get_dropout_layer(dim, p):
    if dim is None:
        dropout = dropouts[f"Dropout"]
    else:
        dropout = dropouts[f"Dropout3d"]
    return dropout(p=p)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"], groups=kwargs['groups'])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)

    def forward(self, data):
        out = self.conv(data)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class ResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ResidBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, kwargs["dim"], groups=kwargs["groups"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            self.downsample = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
            self.norm_res = get_norm(kwargs["norm"], out_channels)

    def forward(self, input_data):
        residual = input_data
        out = self.conv1(input_data)
        out = self.conv2(out)
        out = self.norm(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.norm_res(residual)
        out = self.lrelu(out + residual)
        return out


class DAFTResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 activation='linear', **kwargs):
        super(DAFTResidBlock, self).__init__()
        
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = get_conv(out_channels, out_channels, kernel_size, 1, kwargs["dim"], groups=kwargs["groups"])
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=kwargs["negative_slope"], inplace=True)
        self.downsample = None
        if max(stride) > 1 or in_channels != out_channels:
            self.downsample = get_conv(in_channels, out_channels, kernel_size, stride, kwargs["dim"])
            self.norm_res = get_norm(kwargs["norm"], out_channels)
        
        self.scale = kwargs["scale"]
        self.shift = kwargs["shift"]
        self.ndim_non_img = kwargs["ndim_non_img"]
        self.bottleneck_dim = kwargs["bottleneck_dim"]
        self.location = kwargs["film_location"]
        # self.film_dims = kwargs["film_dims"]
        
        # location decoding
        self.film_dims = 0
        if self.location in {0, 1, 2}:
            self.film_dims = in_channels
        elif self.location in {3, 4}:
            self.film_dims = out_channels
        
        # create aux net feature modulation 
        self.global_pool = nn.AdaptiveAvgPool3d(1)

         # shift and scale decoding
        aux_input_dims = self.film_dims
        self.split_size = 0
        if self.scale and self.shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not self.scale:
            self.scale = 1
            self.shift = None
        elif not self.shift:
            self.shift = 0
            self.scale = None
        layers = [
            ("aux_base", nn.Linear(self.ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift

    def forward(self, feature_map, x_aux):
        
        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)
        
        residual = feature_map
        
        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 2:
            feature_map = self.rescale_features(feature_map, x_aux)
        
        out = self.conv1(feature_map)

        if self.location == 3:
            out = self.rescale_features(out, x_aux)
            
        out = self.conv2(out)
        out = self.norm(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.norm_res(residual)
        out = self.lrelu(out + residual)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.transp_conv = get_transp_conv(in_channels, out_channels, stride, stride, kwargs["dim"])
        self.conv_block = ConvBlock(2 * out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        out = torch.cat((out, skip_data), dim=1)
        out = self.conv_block(out)
        return out


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, input_data):
        return self.conv(input_data)


class NNUNetEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_output_classes: int,
            kernels: List,
            strides: List,
            filters: List,
            normalization_layer: str,
            negative_slope: float,
            residual: bool,
            dimension: int,
            **kwargs
    ):
        super(NNUNetEncoder, self).__init__()
        self.dim = dimension
        self.residual = residual
        self.negative_slope = negative_slope
        self.norm = normalization_layer + f"norm{dimension}d"
        self.filters = filters
        self.in_channels = in_channels

        self.groups_levels = [1] * len(kernels)
        self.groups_levels[:kwargs['depth_wise_conv_levels']] = [-1] * kwargs['depth_wise_conv_levels']

        down_block = ResidBlock if self.residual else ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.in_channels,
            out_channels=self.filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
            groups=self.groups_levels[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
            groups_levels=self.groups_levels[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
        )

        self.dropout = get_dropout_layer(self.dim, kwargs["dropout_rate"])
        self.apply(self.initialize_weights)

        self.pooling = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(filters[-1], num_output_classes)

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride, groups=1, **kwargs):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            negative_slope=self.negative_slope,
            groups=groups,
            **kwargs
        )

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block, groups_levels):
        layers = []
        for in_channel, out_channel, kernel, stride, groups in \
                zip(in_channels, out_channels, kernels, strides, groups_levels):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride, groups)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)

    def forward(self, input_data):
        out = self.input_block(input_data)
        for downsample in self.downsamples:
            out = downsample(out)
        out = self.bottleneck(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class NNUNetEncoderMCM(NNUNetEncoder):
    def __init__(self,
            *args,
            **kwargs,
    ):
        super(NNUNetEncoderMCM, self).__init__(*args, **kwargs,)
        bottleneck_dim = kwargs['bottleneck_dim']
        ndim_non_img = kwargs['ndim_non_img']
        self.mlp = nn.Linear(ndim_non_img, bottleneck_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(kwargs['filters'][-1] + bottleneck_dim, kwargs['num_output_classes'])

    def forward(self, input_data):
        image, tabular = input_data 
        
        out = self.input_block(image)
        for downsample in self.downsamples:
            out = downsample(out)
        out = self.bottleneck(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        
        tab_transformed = self.mlp(tabular)
        tab_transformed = self.relu(tab_transformed)
        out = torch.cat((out, tab_transformed), dim=1)
        out = self.classifier(out)
        return out


class NNUNetEncoder1FC(NNUNetEncoder):
    def __init__(self,
            *args,
            **kwargs,
    ):
        ndim_non_img = kwargs['ndim_non_img']
        super(NNUNetEncoder1FC, self).__init__(*args, **kwargs,)
        self.classifier = nn.Linear(kwargs['filters'][-1] + ndim_non_img, kwargs['num_output_classes'])
        
    def forward(self, input_data):
        image, tabular = input_data 
        
        out = self.input_block(image)
        for downsample in self.downsamples:
            out = downsample(out)
        out = self.bottleneck(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, tabular), dim=1)
        out = self.classifier(out)
        return out


class NNUNetEncoder2FC(NNUNetEncoder):
    def __init__(self,
            *args,
            **kwargs,
    ):  
        
        bottleneck_dim = kwargs['bottleneck_dim']
        ndim_non_img = kwargs['ndim_non_img']
        
        super(NNUNetEncoder2FC, self).__init__(*args, **kwargs,)
        self.mlp = nn.Linear(ndim_non_img, bottleneck_dim)
        self.relu = nn.ReLU()
        layers = [
            ("fc1", nn.Linear(kwargs['filters'][-1] + ndim_non_img, bottleneck_dim)),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(bottleneck_dim, kwargs['num_output_classes'])),
        ]
        self.classifier = nn.Sequential(OrderedDict(layers))
        
        
    def forward(self, input_data):
        image, tabular = input_data 
        
        out = self.input_block(image)
        for downsample in self.downsamples:
            out = downsample(out)
        out = self.bottleneck(out)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = torch.cat((out, tabular), dim=1)
        out = self.classifier(out)
        return out


class NNUNetEncoderDAFT(NNUNetEncoder):
    def __init__(self,
            *args,
            **kwargs,
    ):
        super(NNUNetEncoderDAFT, self).__init__(*args, **kwargs)
        
        down_block = ResidBlock if self.residual else ConvBlock
        fusion_block = DAFTResidBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.in_channels,
            out_channels=kwargs["filters"][0],
            kernel_size=kwargs["kernels"][0],
            stride=kwargs["strides"][0],
            groups=self.groups_levels[0],
        )
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=kwargs["filters"][:-1],
            out_channels=kwargs["filters"][1:],
            kernels=kwargs["kernels"][1:-1],
            strides=kwargs["strides"][1:-1],
            groups_levels=self.groups_levels[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=fusion_block,
            in_channels=kwargs["filters"][-2],
            out_channels=kwargs["filters"][-1],
            kernel_size=kwargs["kernels"][-1],
            stride=kwargs["strides"][-1],
            
            ndim_non_img=kwargs["ndim_non_img"],
            bottleneck_dim=kwargs["bottleneck_dim"], 
            film_location=kwargs["film_location"],
            scale=True, 
            shift=True, 
        )
        
        self.classifier = nn.Linear(kwargs['filters'][-1], kwargs['num_output_classes'])
        
    def forward(self, input_data):
        image, tabular = input_data 
        
        out = self.input_block(image)
        for downsample in self.downsamples:
            out = downsample(out)
        out = self.bottleneck(out, tabular)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out