import os
import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from PSP.pspnet import PSPNet

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)

def load_PSP(device='cpu', backend='densenet'):
    classes = np.array(('Background',  # always index 0 
                        'Hat',          'Hair',      'Glove',     'Sunglasses',
                        'UpperClothes', 'Dress',     'Coat',      'Socks',
                        'Pants',        'Jumpsuits', 'Scarf',     'Skirt',
                        'Face',         'Left-arm',  'Right-arm', 'Left-leg',
                        'Right-leg',    'Left-shoe', 'Right-shoe', ))
    colormap = [(0,0,0),
                (1,0.25,0), (0,0.25,0),  (0.5,0,0.25),   (1,1,1),
                (1,0.75,0), (0,0,0.5),   (0.5,0.25,0),   (0.75,0,0.25),
                (1,0,0.25), (0,0.5,0),   (0.5,0.5,0),    (0.25,0,0.5),
                (1,0,0.75), (0,0.5,0.5), (0.25,0.5,0.5), (1,0,0),
                (1,0.25,0), (0,0.75,0),  (0.5,0.75,0),                 ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    models = {
        'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
        'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
        'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
        'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
        'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
        'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
        'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
    }

    filename = '/home/haoyu/re_id/ReID-master/PSP/{}/PSPNet_last'.format(backend)

    net = models[backend]()
    net = WrappedModel(net)
    # net = nn.DataParallel(net)
    checkpoint = torch.load(filename, map_location=torch.device(device))
    net.load_state_dict(checkpoint)
    net = net.to(device)
    logging.info("loading from {}".format(filename))

    net.eval()

    return net

    """
    pred_seg = pred_seg.argmax(dim=0)
    with torch.no_grad():
        for index, img in enumerate(imgs):
            img = img.unsqueeze(0)
            pred_seg, pred_cls = net(img.to(args.device))
            pred_seg = pred_seg[0]
            pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256))

            plt.imshow(pred, cmap=cmap, norm=norm)
            plt.savefig('{}.png'.format(index))
            
            print(' %d / %d ' % (index, len(imgs)))
    """