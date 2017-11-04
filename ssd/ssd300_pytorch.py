"""
File: ssd300.py
Author: Duino
Email: 472365351duino@gmail.com
Github: github.com/duinodu
Description: model definition of SSD300
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import os, glob

class SSD300(nn.Module):

    def __init__(self, num_classes, phase, pretrain=False):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.phase = phase
        
        self.base_net = self._base_net()
        self.extra_net = self._extra_net()
        self.loc_pred, self.cls_pred = self._predict_net()

        self.priors = None # TODO

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
            pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
            conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
            layers += [pool5, conv6, nn.ReLU(inplace=True),
                              conv7, nn.ReLU(inplace=True)]
            return layers

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 
               512, 512, 512, 'M', 512, 512, 512]
        return nn.ModuleList(make_layers(cfg))

    def _extra_net(self):
        """Extra layers in SSD300, conv8,9,10,11
        Refer https://arxiv.org/pdf/1512.02325.pdf

        Returns:
            extra_net: (ModuleList)
        """
        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 1024
            flag = False

            for i, v in enumerate(cfg):
                if in_channels == 'S':
                    in_channels = v
                    continue
                _kerner_size = (1,3)[flag]
                if v == 'S':
                    conv = nn.Conv2d(in_channels, cfg[i+1],
                                     _kerner_size, stride=2, padding=1)
                else:
                    conv = nn.Conv2d(in_channels, v, _kerner_size)
                layers += [conv]
                in_channels = v
                flag = not flag
            return layers

        cfg = [256, 'S', 512, 
               128, 'S', 256, 
               128, 256, 
               128, 256]
        return nn.ModuleList(make_layers(cfg))

    def _predict_net(self):
        """Predict layer, cls and loc

        Returns:
            loc_layers:  [list], len=6
            conf_layers: [list], len=6

        """
        loc_layers = []
        conf_layers = []
        in_channels = [512, 1024, 512, 256, 256, 256]
        mboxes = [4, 6, 6, 6, 4, 4] # number of boxes per feature map location
        for (in_channels, mbox) in zip(in_channels, mboxes):
            loc_layers += [nn.Conv2d(in_channels, mbox*4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(in_channels, mbox*self.num_classes, kernel_size=3, padding=1)]
        return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

    def forward(self, x):
        """Apply network layers and ops on input image(s) x.

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

        # apply vgg, without BatchNorm
        pred_index = [22,] # conv4_3 relu
        for k, v in enumerate(self.base_net):
            x = v(x)
            if k in pred_index:
                sources.append(x) # without L2Norm
        sources.append(x)

        # apply extra_net and cache source layer outputs
        pred_index = [1,3,5,7]
        for k, v in enumerate(self.extra_net):
            x = v(x) 
            if k in pred_index:
                sources.append(x) 
        
        # apply predict_net to source layers
        for (x, l, c) in zip(sources, self.loc_pred, self.cls_pred):
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
        print('Fetching pretrained model...')
        vgg16 = models.vgg16(pretrained=True)
        model_file = os.path.join(os.environ['HOME'], '.torch/models', 'vgg16-*.pth')
        return glob.glob(model_file)[0]

    def _load_weight(self, weight_file=None):
        """Load pretrained model.
        source: features.[0-28].[weight,bias], classifier.[0,3,6].[weight,bias]
        target: base_net.[0-28].[weight,bias], base_net.[31,33].[weight,bias], -> (load pretrained model) 
                extra_net.[0-7].[xx], loc_pred.[0-5].[xx], cls_pred.[0-5].[xx] -> (init)

        Kwargs:
            weight_file (str): *.pth file path

        Returns: None

        """

        if weight_file == None:
            weight_file = self._fetch_weight()

        _, ext = os.path.splitext(weight_file)

        def downsample(fc, layer):
            """
            downsample weight and bias in fc6,fc7 to conv6,conv7
            w: [512,7,7,4096] -> [512,3,3,1024]     fc6
               [4096, 4096] -> [1024, 1, 1, 1024]   fc7
            b: [4096] -> [1024]
            """
            fc = fc.view(4, 1024, -1)[0] # [4096, 512*7*7] -> [4, 1024, -1][0], 
            if fc.size(1) > 1: # weight
                if layer == 'fc6':
                    fc = fc.view(1024, 512, 7, 7)[:, :, 0::3, 0::3]
                elif layer == 'fc7':
                    fc = fc.view(4, 1024, 1024, 1, 1)[0]
            else:
                fc = fc[:,0]
            return fc

        if ext == '.pkl' or '.pth':
            source_dict = torch.load(weight_file)
            # features -> base_net, remove
            target_dict = {}
            for key in source_dict.keys():
                if 'features' in key: # conv1-5
                    target_dict['base_net'+key[8:]] = source_dict[key] 
                elif 'classifier.0' in key: # conv6
                    target_dict['base_net.31'+key[12:]] = downsample(source_dict[key], 'fc6')
                elif 'classifier.3' in key: # conv7
                    target_dict['base_net.33'+key[12:]] = downsample(source_dict[key], 'fc7')
            source_dict = target_dict 
            # add
            for (key, value) in self.state_dict().items():
                if key not in target_dict.keys():
                    target_dict[key] = value
            self.load_state_dict(target_dict)
            print('Loading weight successfully!')
        else:
            print('Sorry, only .pth and .pkl')

if __name__ == "__main__":
    from torch.autograd import Variable
    net = SSD300(21, 'train', pretrain=True)

    x = Variable(torch.randn(1,3,300,300))
    out = net(x)
    print(out[0].size())
    print(out[1].size())

    import IPython
    IPython.embed()
