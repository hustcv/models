#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: east_pytorch.py
Author: duino
Email: 472365351duino@gmail.com
Github: github.com/duinodu
Description: model definition of EAST
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import os, glob

class EAST(nn.Module):
    def __init__(self, phase, pretrain=False):
        super(EAST, self).__init__()
        self.phase = phase

        self.base_net = self._base_net()
        self.up_net = self._up_net()
        self.score_map = nn.Conv2d(64, 1, kernel_size=1)
        self.rbox = nn.Conv2d(64, 5, kernel_size=1)

        self._init_weight()
        if pretrain:
            self._load_weight()
    
    def _base_net(self):
        """Use vgg16 as basenet. 
        Refer https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py.

        Returns:
            basenet: (ModuleList)
        """
        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif v == 'C':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return layers

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 
               512, 512, 512, 'M', 512, 512, 512, 'M']
        return nn.ModuleList(make_layers(cfg))

    def _up_net(self):
        """Upsampling conv 
        Returns: 
            ModuleList

        """
        def make_layers(cfg):
            layers = []
            in_channels = 512 * 2
            for i, v in enumerate(cfg):
                if i < 3:
                    unpool = nn.Upsample(scale_factor=2)
                    conv1 = nn.Conv2d(in_channels, v, kernel_size=1)
                    conv2 = nn.Conv2d(v, v, kernel_size=3, padding=1)
                    layers += [unpool, conv1, nn.ReLU(inplace=True), conv2, nn.ReLU(inplace=True)]
                    in_channels = v * 2 
                else:
                    conv1 = nn.Conv2d(v, v, kernel_size=3, padding=1)
                    layers += [conv1, nn.ReLU(inplace=True)]
            return layers

        cfg = [256, 128, 64, 64]
        return nn.ModuleList(make_layers(cfg))


    def forward(self, x):

        # apply vgg on x and save features
        features = []
        up_index = [9, 16, 23] # where to concatenate feature maps
        for k, v in enumerate(self.base_net):
            x = v(x)
            if k in up_index:
                features.append(x)

        # upsampling
        features = features[::-1]
        for ind in range(len(features)): 
            x = self.up_net[ind*5](x) # upsampling
            x = torch.cat((x, features[ind]), 1)
            for i in range(1,5):
                x = self.up_net[ind*5+i](x)
        x = self.up_net[-2](x)
        x = self.up_net[-1](x)

        # pred
        conf = self.score_map(x)
        rbox = self.rbox(x)
        return (conf, rbox)

    def _init_weight(self):
        def weight_init(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.apply(weight_init)

    def _fetch_weight(self):
        """Fetch pretrain model using torchvision.
        Returns: 
            weight_file: (str) pretrained weight file path

        """
        print('fetching pretrained model...')
        model_file = os.path.join(os.environ['HOME'], '.torch/models', 'vgg16-*.pth')
        ret = glob.glob(model_file)
        if len(ret) == 0:
            vgg16 = models.vgg16(pretrained=True)
            ret = glob.glob(model_file)
        return ret[0]


    def _load_weight(self, weight_file=None):
        """Load pretrained model

        Kwargs:
            weight_file (str): *.pth file path

        Returns: None

        """

        if weight_file == None:
            weight_file = self._fetch_weight()

        _, ext = os.path.splitext(weight_file)

        if ext == '.pkl' or '.pth':
            saved_state_dict = torch.load(weight_file)

            # features -> base_net, remove
            saved_state_dict2 = {}
            for key in saved_state_dict.keys():
                if 'features' in key:
                    saved_state_dict2['base_net'+key[8:]] = saved_state_dict[key] 
            saved_state_dict = saved_state_dict2 
            # add
            for (key, value) in self.state_dict().items():
                if key not in saved_state_dict.keys():
                    saved_state_dict[key] = value
            self.load_state_dict(saved_state_dict)
            print('Loading weight successfully!')
        else:
            print('Sorry, only .pth and .pkl')

if __name__ == "__main__":
    from torch.autograd import Variable
    net = EAST('train', True)

    x = Variable(torch.randn(1,3,512,512))
    out = net(x)
    print(out[0].size())
    print(out[1].size())


    import IPython
    IPython.embed()
