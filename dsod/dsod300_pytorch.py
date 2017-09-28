#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: dsod300_pytorch.py
Author: duino
Email: 472365351duino@gmail.com
Github: github.com/duinodu
Description: 
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import os, glob

# https://github.com/pytorch/vision/tree/master/torchvision/models/densenet.py

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, withPooling=True):

        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_input_features,
                                          kernel_size=1, stride=1, bias=False))
        if withPooling:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))

class _ExtraBlock(nn.Sequential):
    """Dense Prediction Structure, plain branch is as SSD does
    """
    def __init__(self, in_channels, out_channels, stride, ceil_mode):
        super(_ExtraBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
        if stride == 2:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) 
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1) 
    
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x2 = self.pool(x)
        x2 = F.relu(self.conv3(x2))
        return torch.cat((x1, x2), 1)


class DSOD300(nn.Module):
    def __init__(self, num_classes, phase, densenet_type='non_pretrain', pretrain=False):
        """Init of DSOD300.

        Args:
            num_classes (int): including _bg_, such as 21 for voc2007
            phase (str): 'train', 'test'

        Kwargs:
            densenet_type (str): 'non_pretrain', 'pretrain' for different base net
            pretrain (bool): whether to load pretrained model 

        Returns: None

        """
        super(DSOD300, self).__init__()
        self.num_classes = num_classes
        self.phase = phase

        if densenet_type == 'pretrain':
            raise NotImplementedError

        self.densenet_type = densenet_type
        self.base_net = self._base_net(self.densenet_type)
        self.extra_net = self._extra_net()
        self.loc_pred, self.conf_pred = self._predict_net()

        self.priors = None

        self._init_weight()
        if pretrain:
            if self.densenet_type == 'pretrain':
                self._load_weight()
            else:
                print("In non pretrain mode")

    def _base_net(self, densenet_type):
        """Using DenseNet as backbone network.

        Args:
            in_channels (int): number of input channels
            densenet_type (str):
                'pretrain': block_config = (6,12,32,24), densenet161
                'non-pretrain': block_conifg = (6,8,8,8)

        Returns: 
            self.features
        """
        in_channels = 3

        # stem
        layers = nn.Sequential()

        stemblock = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2))
        layers.add_module('stemblock', stemblock)

        num_features = 128
        bn_size = 4
        growth_rate = 48
        drop_rate = 0
        if densenet_type == 'pretrain':
            block_config = (6, 12, 32, 24)
        else:
            block_config = (6, 8, 8, 8)

        for i, num_layers in enumerate(block_config):
            denseblock = _DenseBlock(num_layers=num_layers, num_input_features=num_features, 
                                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            num_features = num_features + num_layers * growth_rate
            transition = _Transition(num_input_features=num_features, withPooling=(i<2))
            layers.add_module('denseblock%d' % (i+1), denseblock)
            layers.add_module('transition%d' % (i+1), transition)

        # Final batch norm, as DenseNet did
        layers.add_module('norm5', nn.BatchNorm2d(num_features))
        return layers

    def _extra_net(self):
        """Extra layers in DSOD300.

        Returns:
            extra_net: (ModuleList)
        """
        layers = []
        in_channels = 1568

        cfg = [256, 128, 128, 128]
        stride = [2,2,1,1]
        ceil_mode = [True, True, True, False]
        for (v,s,c) in zip(cfg, stride, ceil_mode):
            block = _ExtraBlock(in_channels, out_channels=v, stride=s, ceil_mode=c)
            layers += [block]
            in_channels = v*2
        return nn.ModuleList(layers)

    def _predict_net(self):
        """Predict layer, cls and loc

        Returns:
            loc_layers:  [list], len=6
            conf_layers: [list], len=6

        """
        loc_layers = []
        conf_layers = []
        in_channels = [800, 1568, 512, 256, 256, 256]
        mboxes = [4, 6, 6, 6, 4, 4] # number of boxes per feature map location
        for (in_channels, mbox) in zip(in_channels, mboxes):
            loc_layers += [nn.Conv2d(in_channels, mbox*4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(in_channels, mbox*self.num_classes, kernel_size=3, padding=1)]
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    def forward(self, x):
        """Apply network layers and ops on input image(s) x.
        Dense Connection Prediction is implemented here.

        Args:
            x (tensor): input image or batch of image.
                Shape: [batch, 3, 300, 300]

        Returns: 
            Depending on phase;
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors*4]
                    3: priorbox layers, Shape: [2, num_priors*4]
            test:
                Variale(tensor) of input class label predictions
                ..
        """
        sources = [] # feature maps where to make predictions 
        conf = []
        loc = []

        # apply dense_net
        pred_index = [3,] # dense block 2

        for k, v in enumerate(self.base_net):
            x = v(x)
            if k in pred_index:
                sources.append(x) # 38
        sources.append(x) # 19

        # apply extra_net and cache source layer outputs
        for k, v in enumerate(self.extra_net):
            x = v(x) 
            sources.append(x) 

        # apply predict_net to source layers
        for (x, l, c) in zip(sources, self.loc_pred, self.conf_pred):
            loc.append(l(x).permute(0,2,3,1).contiguous()) # [B,C,H,W] -> [B,H,W,C]
            conf.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) # to concat pred from many layers
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'train':
            output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(loc.size(0), -1, self.num_classes),
                    self.priors
            )
        else:
            raise NotImplementedError, "test phase"
        return output

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
        densenet161 = models.densenet161(pretrained=True)
        model_file = os.path.join(os.environ['HOME'], '.torch/models', 'densenet161-*.pth')
        return glob.glob(model_file)[0]

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
                if 'features.denseblock' in key or \
                   'features.transition' in key:
                    if 'conv' in key: # only load conv parameters
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
    #net = DSOD300(21, 'train')
    net = DSOD300(21, 'train', 'pretrain', True)

    x = Variable(torch.randn(1,3,300,300))
    out = net(x)

    import IPython
    IPython.embed()

