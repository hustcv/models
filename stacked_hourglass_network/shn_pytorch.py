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
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes/2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes/2)
        self.conv2 = nn.Conv2d(planes/2, planes/2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes/2)
        self.conv3 = nn.Conv2d(planes/2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.numIn = inplanes
        self.numOut = planes
        if self.numIn != self.numOut:
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

        if self.numIn != self.numOut:
            residual = self.conv_res(residual)
            residual = self.bn_res(residual)

        out += residual
        #out = self.relu(out)
        return out

class Hourglass(nn.Module):

    """Hourglass module"""

    def __init__(self, numIn, numOut):
        super(Hourglass, self).__init__()
        # upper branch
        self.hat1 = self._make_hat(numIn, numOut)
        self.hat2 = self._make_hat(256, 256)
        self.hat3 = self._make_hat(256, 256)
        self.hat4 = self._make_hat(256, 256)

        # lower branch
        self.left1 = self._make_left(numIn, 256)
        self.left2 = self._make_left(256, 256)
        self.left3 = self._make_left(256, 256)
        self.left4 = self._make_left(256, 256)
        self.right4 = self._make_right(256, 256)
        self.right3 = self._make_right(256, 256)
        self.right2 = self._make_right(256, 256)
        self.right1 = self._make_right(256, numOut)
        self.middle = Residual(256, 256)

    def _make_hat(self, numIn, numOut):
        layers = []
        layers.append(Residual(numIn, 256))
        layers.append(Residual(256, 256))
        layers.append(Residual(256, numOut))
        return nn.Sequential(*layers)

    def _make_left(self, numIn, numOut):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        layers.append(Residual(numIn, 256))
        layers.append(Residual(256, 256))
        layers.append(Residual(256, numOut))
        return nn.Sequential(*layers)

    def _make_right(self, numIn, numOut):
        layers = []
        layers.append(Residual(numIn, numOut))
        #layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        layers.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layers)

    def element_add(self, x, elem):
        dst_h = x.size(-2) 
        dst_w = x.size(-1)
        if dst_h > elem.size(-2):
            dst_h = elem.size(-2)
        if dst_w > elem.size(-1):
            dst_w = elem.size(-1)
        elif x.size(-1) < elem.size(-1) or x.size(-2) < elem.size(-1):
            raise NotImplementedError
        return x[:,:,:dst_h, :dst_w] + elem

    def forward(self, x):
        h1 = self.hat1(x) # 75
        x = self.left1(x) # 75 -> 38 
        h2 = self.hat2(x) # 38
        x = self.left2(x) # 38 -> 19
        h3 = self.hat3(x) # 19
        x = self.left3(x) # 19 -> 10
        h4 = self.hat4(x) # 10
        x = self.left4(x) # 10 -> 5

        x = self.middle(x)

        x = self.element_add(self.right4(x), h4) 
        x = self.element_add(self.right3(x), h3) 
        x = self.element_add(self.right2(x), h2) 
        x = self.element_add(self.right1(x), h1) 
        return x


class SHN(nn.Module):

    def __init__(self, numOut=14):
        super(SHN, self).__init__()
        self.numOut = numOut

        # head
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = Residual(64, 128)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.res2_1 = Residual(128, 128)
        self.res2_2 = Residual(128, 128)
        self.res2_3 = Residual(128, 256)

        self.hg1 = Hourglass(256, 512)
        self.hg2 = Hourglass(256+128, 512)

        self.linear1 = self._make_conv1x1s(512, 256)
        self.linear2 = self._make_conv1x1s(512, 512)

        self.conv2 = nn.Conv2d(self.numOut, 256+128, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(256+128, 256+128, kernel_size=1, stride=1, padding=0, bias=False)

        self.predict1 = nn.Conv2d(256, self.numOut, kernel_size=1, stride=1, padding=0, bias=False)
        self.predict2 = nn.Conv2d(512, self.numOut, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_conv1x1s(self, numIn, numOut):
        layers = []
        layers.append(nn.Conv2d(numIn, 256, kernel_size=1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, numOut, kernel_size=1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(numOut))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x) # 300 -> 150
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        p1 = self.maxpool(x) # 150 -> 75
        x = self.res2_1(p1)
        x = self.res2_2(x)
        x = self.res2_3(x)

        x = self.hg1(x) # 150
        x = self.linear1(x)
        pred1 = self.predict1(x)

        x = torch.cat((x, p1), 1)
        x = self.conv2(pred1) + self.conv3(x)

        x = self.hg2(x)
        x = self.linear2(x)
        pred2 = self.predict2(x)
        
        return pred1, pred2

    def init_weight(self, weight_file=None):
        pass

if __name__ == "__main__":
    from torch.autograd import Variable
    net = SHN(2)

    x = Variable(torch.randn(1,3,1000,800))
    pred1, pred2 = net(x)
    print(pred1.size())
    print(pred2.size())

    import IPython
    IPython.embed()

