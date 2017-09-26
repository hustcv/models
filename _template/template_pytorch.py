#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: _template_pytorch.py
Author: xxx
Email: xxx
Github: github.com/xxx
Description: xxx
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class name(nn.Module):
    def __init__(self, pretrain=True):
        super(name, self).__init__()
        # TODO: design model


        self._init_weight()
        if pretrain:
            self._load_weight()

    def forward(self, x):
        # TODO: build model
        return x

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
        # TODO: change to your own basenet name
        vgg16 = models.vgg16(pretrained=True)
        model_file = os.path.join(os.environ['HOME'], '.torch/models', 'vgg16-*.pth')
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
            # TODO: change 'features'
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
    # TODO: test model
    pass
