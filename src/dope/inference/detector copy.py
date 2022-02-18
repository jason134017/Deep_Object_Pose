# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

'''
Contains the following classes:
   - ModelData - High level information encapsulation
   - ObjectDetector - Greedy algorithm to build cuboids from belief maps 
'''

import time
from os import path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from torch.autograd import Variable

from dope.utils import get_image_grid

# Import the definition of the neural network model and cuboids

#global transform for image input
transform = transforms.Compose([
    # transforms.Scale(IMAGE_SIZE),
    # transforms.CenterCrop((imagesize,imagesize)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


#================================ Models ================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )
        
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DopeNetwork(nn.Module):
    def __init__(
            self,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = models.vgg19(pretrained=False).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model

class DopeMobileNet(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeMobileNet, self).__init__()

        #self.mobile_feature = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features
        self.mobile_feature = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).features
        # upsample to 50x50 from 13x13
        self.upsample = nn.Sequential()
        self.upsample.add_module('0', nn.Upsample(scale_factor=2))

        # should this go before the upsample?
        # self.upsample.add_module('4', nn.Conv2d(1280, 640,
        #     kernel_size=3, stride=1, padding=1))
        self.upsample.add_module('44',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        # self.upsample.add_module('55',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # self.upsample.add_module('5', nn.ReLU(inplace=True))

        # self.upsample.add_module('6', nn.Conv2d(640, 320,
        #     kernel_size=3, stride=1, padding=1))

        self.upsample.add_module('10', nn.Upsample(scale_factor=2))
        # self.upsample.add_module('14', nn.Conv2d(320, 160,
        #     kernel_size=3, stride=1, padding=1))
        # self.upsample.add_module('15', nn.ReLU(inplace=True))
        # self.upsample.add_module('16', nn.Conv2d(160, 64,
        #     kernel_size=3, stride=1, padding=0))
        self.upsample.add_module('55',InvertedResidual(640, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        self.upsample.add_module('56',InvertedResidual(320, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # set 50,50
        self.upsample.add_module('4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        # final output - change that for mobile block
        # self.heads_0 = nn.Sequential()

        def build_block(inputs, outputs, nb_layers = 2 ):
            layers = []
            layers.append(InvertedResidual(inputs, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
            for l in range(nb_layers-1):
                layers.append(InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))        
            layers.append(nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            # layers.append('4', nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        self.head_0_beliefs = build_block(64,numBeliefMap)
        self.head_0_aff = build_block(64,(numBeliefMap-1)*2,3)

        self.head_1_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_1_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,2)

        self.head_2_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_2_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,1)



    def forward(self, x):
        '''Runs inference on the neural network'''
        # print(x.shape)
        out_features = self.mobile_feature(x)
        # print('out2_features',out_features.shape)
        output_up = self.upsample(out_features)
        # print('output_up',output_up.shape)

        # stages
        belief_0 = self.head_0_beliefs(output_up)
        aff_0 = self.head_0_aff(output_up)

        # print(belief_0.shape)

        out_0 = torch.cat([output_up, belief_0, aff_0], 1)

        # print(out_0.shape)
        # raise()
        belief_1 = self.head_1_beliefs(out_0)
        aff_1 = self.head_1_aff(out_0)

        out_1 = torch.cat([output_up, belief_1, aff_1], 1)

        belief_2 = self.head_2_beliefs(out_1)
        aff_2 = self.head_2_aff(out_1)

        return  [belief_0,belief_1,belief_2],\
                [aff_0,aff_1,aff_2]
"""
class DopeMobileNetV3_Large(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeMobileNetV3_Large, self).__init__()

        #self.mobile_feature = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features
        self.mobile_feature =  models.mobilenet_v3_large(pretrained=True).features
        # upsample to 50x50 from 13x13
        self.upsample = nn.Sequential()
        self.upsample.add_module('0', nn.Upsample(scale_factor=2))

        # should this go before the upsample?
        # self.upsample.add_module('4', nn.Conv2d(1280, 640,
        #     kernel_size=3, stride=1, padding=1))
        self.upsample.add_module('44',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        # self.upsample.add_module('55',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # self.upsample.add_module('5', nn.ReLU(inplace=True))

        # self.upsample.add_module('6', nn.Conv2d(640, 320,
        #     kernel_size=3, stride=1, padding=1))

        self.upsample.add_module('10', nn.Upsample(scale_factor=2))
        # self.upsample.add_module('14', nn.Conv2d(320, 160,
        #     kernel_size=3, stride=1, padding=1))
        # self.upsample.add_module('15', nn.ReLU(inplace=True))
        # self.upsample.add_module('16', nn.Conv2d(160, 64,
        #     kernel_size=3, stride=1, padding=0))
        self.upsample.add_module('55',InvertedResidual(640, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        self.upsample.add_module('56',InvertedResidual(320, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # set 50,50
        self.upsample.add_module('4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        # final output - change that for mobile block
        # self.heads_0 = nn.Sequential()

        def build_block(inputs, outputs, nb_layers = 2 ):
            layers = []
            layers.append(InvertedResidual(inputs, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
            for l in range(nb_layers-1):
                layers.append(InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))        
            layers.append(nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            # layers.append('4', nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        self.head_0_beliefs = build_block(64,numBeliefMap)
        self.head_0_aff = build_block(64,(numBeliefMap-1)*2,3)

        self.head_1_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_1_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,2)

        self.head_2_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_2_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,1)



    def forward(self, x):
        '''Runs inference on the neural network'''
        # print(x.shape)
        out_features = self.mobile_feature(x)
        # print('out2_features',out_features.shape)
        output_up = self.upsample(out_features)
        # print('output_up',output_up.shape)

        # stages
        belief_0 = self.head_0_beliefs(output_up)
        aff_0 = self.head_0_aff(output_up)

        # print(belief_0.shape)

        out_0 = torch.cat([output_up, belief_0, aff_0], 1)

        # print(out_0.shape)
        # raise()
        belief_1 = self.head_1_beliefs(out_0)
        aff_1 = self.head_1_aff(out_0)

        out_1 = torch.cat([output_up, belief_1, aff_1], 1)

        belief_2 = self.head_2_beliefs(out_1)
        aff_2 = self.head_2_aff(out_1)

        return  [belief_0,belief_1,belief_2],\
                [aff_0,aff_1,aff_2]   
"""     
"""
class DopeMobileNetV3_Small(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeMobileNetV3_Small, self).__init__()

        #self.mobile_feature = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).features
        self.mobile_feature =  models.mobilenet_v3_small(pretrained=True).features
        # upsample to 50x50 from 13x13
        self.upsample = nn.Sequential()
        self.upsample.add_module('0', nn.Upsample(scale_factor=2))

        # should this go before the upsample?
        # self.upsample.add_module('4', nn.Conv2d(1280, 640,
        #     kernel_size=3, stride=1, padding=1))
        self.upsample.add_module('44',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        # self.upsample.add_module('55',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # self.upsample.add_module('5', nn.ReLU(inplace=True))

        # self.upsample.add_module('6', nn.Conv2d(640, 320,
        #     kernel_size=3, stride=1, padding=1))

        self.upsample.add_module('10', nn.Upsample(scale_factor=2))
        # self.upsample.add_module('14', nn.Conv2d(320, 160,
        #     kernel_size=3, stride=1, padding=1))
        # self.upsample.add_module('15', nn.ReLU(inplace=True))
        # self.upsample.add_module('16', nn.Conv2d(160, 64,
        #     kernel_size=3, stride=1, padding=0))
        self.upsample.add_module('55',InvertedResidual(640, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        self.upsample.add_module('56',InvertedResidual(320, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # set 50,50
        self.upsample.add_module('4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        # final output - change that for mobile block
        # self.heads_0 = nn.Sequential()

        def build_block(inputs, outputs, nb_layers = 2 ):
            layers = []
            layers.append(InvertedResidual(inputs, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
            for l in range(nb_layers-1):
                layers.append(InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))        
            layers.append(nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            # layers.append('4', nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        self.head_0_beliefs = build_block(64,numBeliefMap)
        self.head_0_aff = build_block(64,(numBeliefMap-1)*2,3)

        self.head_1_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_1_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,2)

        self.head_2_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_2_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,1)



    def forward(self, x):
        '''Runs inference on the neural network'''
        # print(x.shape)
        out_features = self.mobile_feature(x)
        # print('out2_features',out_features.shape)
        output_up = self.upsample(out_features)
        # print('output_up',output_up.shape)

        # stages
        belief_0 = self.head_0_beliefs(output_up)
        aff_0 = self.head_0_aff(output_up)

        # print(belief_0.shape)

        out_0 = torch.cat([output_up, belief_0, aff_0], 1)

        # print(out_0.shape)
        # raise()
        belief_1 = self.head_1_beliefs(out_0)
        aff_1 = self.head_1_aff(out_0)

        out_1 = torch.cat([output_up, belief_1, aff_1], 1)

        belief_2 = self.head_2_beliefs(out_1)
        aff_2 = self.head_2_aff(out_1)

        return  [belief_0,belief_1,belief_2],\
                [aff_0,aff_1,aff_2]        
"""
"""
class DopeEfficientNet(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeEfficientNet, self).__init__()

        self.efficientNet_feature = models.efficientnet_b7(pretrained=True).features

        # upsample to 50x50 from 13x13
        self.upsample = nn.Sequential()
        self.upsample.add_module('0', nn.Upsample(scale_factor=2))

        # should this go before the upsample?
        # self.upsample.add_module('4', nn.Conv2d(1280, 640,
        #     kernel_size=3, stride=1, padding=1))
        self.upsample.add_module('44',InvertedResidual(2560, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        # self.upsample.add_module('55',InvertedResidual(1280, 640, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # self.upsample.add_module('5', nn.ReLU(inplace=True))

        # self.upsample.add_module('6', nn.Conv2d(640, 320,
        #     kernel_size=3, stride=1, padding=1))

        self.upsample.add_module('10', nn.Upsample(scale_factor=2))
        # self.upsample.add_module('14', nn.Conv2d(320, 160,
        #     kernel_size=3, stride=1, padding=1))
        # self.upsample.add_module('15', nn.ReLU(inplace=True))
        # self.upsample.add_module('16', nn.Conv2d(160, 64,
        #     kernel_size=3, stride=1, padding=0))
        self.upsample.add_module('55',InvertedResidual(640, 320, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
        self.upsample.add_module('56',InvertedResidual(320, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))

        # set 50,50
        self.upsample.add_module('4', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        # final output - change that for mobile block
        # self.heads_0 = nn.Sequential()

        def build_block(inputs, outputs, nb_layers = 2 ):
            layers = []
            layers.append(InvertedResidual(inputs, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))
            for l in range(nb_layers-1):
                layers.append(InvertedResidual(64, 64, stride=1, expand_ratio=6, norm_layer=nn.BatchNorm2d))        
            layers.append(nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            # layers.append('4', nn.Conv2d(64, outputs, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        self.head_0_beliefs = build_block(64,numBeliefMap)
        self.head_0_aff = build_block(64,(numBeliefMap-1)*2,3)

        self.head_1_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_1_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,2)

        self.head_2_beliefs = build_block(64+numBeliefMap+((numBeliefMap-1)*2),numBeliefMap,3)
        self.head_2_aff = build_block(64+numBeliefMap+(numBeliefMap-1)*2,(numBeliefMap-1)*2,1)



    def forward(self, x):
        '''Runs inference on the neural network'''
        # print(x.shape)
        out_features = self.efficientNet_feature(x)
        # print('out2_features',out_features.shape)
        output_up = self.upsample(out_features)
        # print('output_up',output_up.shape)

        # stages
        belief_0 = self.head_0_beliefs(output_up)
        aff_0 = self.head_0_aff(output_up)

        # print(belief_0.shape)

        out_0 = torch.cat([output_up, belief_0, aff_0], 1)

        # print(out_0.shape)
        # raise()
        belief_1 = self.head_1_beliefs(out_0)
        aff_1 = self.head_1_aff(out_0)

        out_1 = torch.cat([output_up, belief_1, aff_1], 1)

        belief_2 = self.head_2_beliefs(out_1)
        aff_2 = self.head_2_aff(out_1)

        return  [belief_0,belief_1,belief_2],\
                [aff_0,aff_1,aff_2]
"""

class ModelData(object):
    '''This class contains methods for loading the neural network'''

    def __init__(self, name="", net_path="", gpu_id=0):
        self.name = name
        self.net_path = net_path  # Path to trained network model
        self.net = None  # Trained network
        self.gpu_id = gpu_id

    def get_net(self):
        '''Returns network'''
        if not self.net:
            self.load_net_model()
        return self.net

    def load_net_model(self):
        '''Loads network model from disk'''
        if not self.net and path.exists(self.net_path):
            self.net = self.load_net_model_path(self.net_path)
        if not path.exists(self.net_path):
            print("ERROR:  Unable to find model weights: '{}'".format(
                self.net_path))
            exit(0)

    def load_net_model_path(self, path):
        '''Loads network model from disk with given path'''
        model_loading_start_time = time.time()
        print("Loading DOPE model '{}'...".format(path))
        net = DopeNetwork()
        net = torch.nn.DataParallel(net, [0]).cuda()
        net.load_state_dict(torch.load(path))
        net.eval()
        print('    Model loaded in {} seconds.'.format(
            time.time() - model_loading_start_time))
        return net

    def __str__(self):
        '''Converts to string'''
        return "{}: {}".format(self.name, self.net_path)


#================================ ObjectDetector ================================
class ObjectDetector(object):
    '''This class contains methods for object detection'''

    @staticmethod
    def detect_object_in_image(net_model, pnp_solver, in_img, config,
                               make_belief_debug_img=False, norm_belief=True, overlay_image=True):
        """
        Detect objects in a image using a specific trained network model
        Returns the poses of the objects and the belief maps
        """
        if in_img is None:
            return []

        # Run network inference
        image_tensor = transform(in_img)
        image_torch = Variable(image_tensor).cuda().unsqueeze(0)
        out, seg = net_model(image_torch)
        vertex2 = out[-1][0]
        aff = seg[-1][0]

        # Find objects from network output
        detected_objects = ObjectDetector.find_object_poses(vertex2, aff, pnp_solver, config)

        if not make_belief_debug_img:
            return detected_objects, None
        else:
            # Run the belief maps debug display on the belief maps
            tensor = vertex2
            belief_imgs = []
            if overlay_image:
                upsampling = nn.UpsamplingNearest2d(size=in_img.shape[:2])
                in_img = (torch.tensor(in_img).float() / 255.0)
                in_img *= 0.5

            for j in range(tensor.size()[0]):
                belief = tensor[j].clone()
                if norm_belief:
                    belief -= float(torch.min(belief).item())
                    belief /= float(torch.max(belief).item())

                belief = torch.clamp(belief, 0, 1).cpu()
                if overlay_image:
                    belief = upsampling(belief.unsqueeze(0).unsqueeze(0)).squeeze().squeeze().data
                    belief = torch.cat([
                        belief.unsqueeze(0) + in_img[:, :, 0],
                        belief.unsqueeze(0) + in_img[:, :, 1],
                        belief.unsqueeze(0) + in_img[:, :, 2]
                    ]).unsqueeze(0)
                    belief = torch.clamp(belief, 0, 1)
                else:
                    belief = torch.cat([
                        belief.unsqueeze(0),
                        belief.unsqueeze(0),
                        belief.unsqueeze(0)
                    ]).unsqueeze(0)
                belief_imgs.append(belief.data.squeeze().numpy())

            # Create the image grid
            belief_imgs = torch.tensor(np.array(belief_imgs))
            im_belief = get_image_grid(belief_imgs, mean=0, std=1)

            return detected_objects, im_belief

    @staticmethod
    def find_object_poses(vertex2, aff, pnp_solver, config):
        '''Detect objects given network output'''

        # Detect objects from belief maps and affinities
        objects, all_peaks = ObjectDetector.find_objects(vertex2, aff, config)
        detected_objects = []
        obj_name = pnp_solver.object_name

        for obj in objects:
            # Run PNP
            points = obj[1] + [(obj[0][0]*8, obj[0][1]*8)]
            cuboid2d = np.copy(points)
            location, quaternion, projected_points = pnp_solver.solve_pnp(points)

            # Save results
            detected_objects.append({
                'name': obj_name,
                'location': location,
                'quaternion': quaternion,
                'cuboid2d': cuboid2d,
                'projected_points': projected_points,
                'score': obj[-1],
            })

        return detected_objects

    @staticmethod
    def find_objects(vertex2, aff, config, numvertex=8):
        '''Detects objects given network belief maps and affinities, using heuristic method'''

        all_peaks = []
        peak_counter = 0
        for j in range(vertex2.size()[0]):
            belief = vertex2[j].clone()
            map_ori = belief.cpu().data.numpy()

            map = gaussian_filter(belief.cpu().data.numpy(), sigma=config.sigma)
            p = 1
            map_left = np.zeros(map.shape)
            map_left[p:,:] = map[:-p,:]
            map_right = np.zeros(map.shape)
            map_right[:-p,:] = map[p:,:]
            map_up = np.zeros(map.shape)
            map_up[:,p:] = map[:,:-p]
            map_down = np.zeros(map.shape)
            map_down[:,:-p] = map[:,p:]

            peaks_binary = np.logical_and.reduce(
                                (
                                    map >= map_left,
                                    map >= map_right,
                                    map >= map_up,
                                    map >= map_down,
                                    map > config.thresh_map)
                                )
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])

            # Computing the weigthed average for localizing the peaks
            peaks = list(peaks)
            win = 5
            ran = win // 2
            peaks_avg = []
            for p_value in range(len(peaks)):
                p = peaks[p_value]
                weights = np.zeros((win,win))
                i_values = np.zeros((win,win))
                j_values = np.zeros((win,win))
                for i in range(-ran,ran+1):
                    for j in range(-ran,ran+1):
                        if p[1]+i < 0 \
                                or p[1]+i >= map_ori.shape[0] \
                                or p[0]+j < 0 \
                                or p[0]+j >= map_ori.shape[1]:
                            continue

                        i_values[j+ran, i+ran] = p[1] + i
                        j_values[j+ran, i+ran] = p[0] + j

                        weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

                # if the weights are all zeros
                # then add the none continuous points
                OFFSET_DUE_TO_UPSAMPLING = 0.4395
                try:
                    peaks_avg.append(
                        (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                         np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
                except:
                    peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
            # Note: Python3 doesn't support len for zip object
            peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

            peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

            id = range(peak_counter, peak_counter + peaks_len)

            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += peaks_len

        objects = []

        # Check object centroid and build the objects if the centroid is found
        for nb_object in range(len(all_peaks[-1])):
            if all_peaks[-1][nb_object][2] > config.thresh_points:
                objects.append([
                    [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                    [None for i in range(numvertex)],
                    [None for i in range(numvertex)],
                    all_peaks[-1][nb_object][2]
                ])

        # Working with an output that only has belief maps
        if aff is None:
            if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
                for i_points in range(8):
                    if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > config.threshold:
                        objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
        else:
            # For all points found
            for i_lists in range(len(all_peaks[:-1])):
                lists = all_peaks[i_lists]

                for candidate in lists:
                    if candidate[2] < config.thresh_points:
                        continue

                    i_best = -1
                    best_dist = 10000
                    best_angle = 100
                    for i_obj in range(len(objects)):
                        center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                        # integer is used to look into the affinity map,
                        # but the float version is used to run
                        point_int = [int(candidate[0]), int(candidate[1])]
                        point = [candidate[0], candidate[1]]

                        # look at the distance to the vector field.
                        v_aff = np.array([
                                        aff[i_lists*2,
                                        point_int[1],
                                        point_int[0]].data.item(),
                                        aff[i_lists*2+1,
                                            point_int[1],
                                            point_int[0]].data.item()]) * 10

                        # normalize the vector
                        xvec = v_aff[0]
                        yvec = v_aff[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec/=norms
                        yvec/=norms

                        v_aff = np.concatenate([[xvec],[yvec]])

                        v_center = np.array(center) - np.array(point)
                        xvec = v_center[0]
                        yvec = v_center[1]

                        norms = np.sqrt(xvec * xvec + yvec * yvec)

                        xvec /= norms
                        yvec /= norms

                        v_center = np.concatenate([[xvec],[yvec]])

                        # vector affinity
                        dist_angle = np.linalg.norm(v_center - v_aff)

                        # distance between vertexes
                        dist_point = np.linalg.norm(np.array(point) - np.array(center))

                        if dist_angle < config.thresh_angle and (best_dist > 1000 or best_dist > dist_point):
                            i_best = i_obj
                            best_angle = dist_angle
                            best_dist = dist_point

                    if i_best == -1:
                        continue

                    if objects[i_best][1][i_lists] is None \
                            or best_angle < config.thresh_angle \
                            and best_dist < objects[i_best][2][i_lists][1]:
                        objects[i_best][1][i_lists] = ((candidate[0])*8, (candidate[1])*8)
                        objects[i_best][2][i_lists] = (best_angle, best_dist)

        return objects, all_peaks
