#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: shn_pytorch.py
Author: Duino
Email: 472365351duino@gmail.com
Github: github.com/duinodu
Description: Model definition of Stacked Hourglass Networks
    Ref: https://arxiv.org/pdf/1603.06937.pdf
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Residual(nn.Module):
    """Residual block"""

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes/2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes/2)
        self.conv2 = nn.Conv2d(planes/2, planes/2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes/2)
        self.conv3 = nn.Conv2d(planes/2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if inplanes != planes:
            self.conv_res = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn_res = nn.BatchNorm2d(planes)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if inplanes != planes:
            residual = self.conv_res(x)
            residual = self.bn_res(x)

        out += residual
        #out = self.relu(out)

        return out

class Hourglass(nn.Module):

    """Hourglass module"""

    def __init__(self, n, numIn, numOut):
        super(Hourglass, self).__init__()
        # upper branch
        self.up1 = Residual(numIn, 256)
        self.up2 = Residual(256, 256)
        self.up4 = Residual(256, numOut)
        # lower branch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.low1 = Residual(numIn, 256)
        self.low2 = Residual(256, 256)
        self.low5 = Residual(256, 256)


class shn(nn.Module):

    def __init__(self):
        super(shn, self).__init__()
        # TODO: design model

    def forward(self, x):
        # TODO: build model
        return x

    def init_weight(self, weight_file=None):
        pass

if __name__ == "__main__":
    # TODO: test model
    pass
