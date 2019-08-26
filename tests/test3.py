import torch
import torch.nn as nn

torch.manual_seed(100)

import torch.nn.functional as F
from torch.autograd import Variable

import math

inputs_ = torch.rand((10, 200))
targets = torch.Tensor([11, 11, 11, 12, 13, 12, 13, 13, 12, 13])
eps = 1e-5
m = 0.5

n = inputs_.size(0)
inputs = F.normalize(inputs_)

# arcos distance matrix
dist = torch.FloatTensor(n, n).zero_()#.to('cuda')
for i in range(n):
    dist[i, :] = (inputs[i] * inputs).sum(dim=1)
dist = dist.clamp(min=-1+eps, max=1-eps)
dist = torch.acos(dist) 

# For each anchor, find the hardest positive and negative
mask = targets.expand(n, n).eq(targets.expand(n, n).t())
dist_ap, dist_an = [], []
for i in range(n):
    dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
    dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
dist_ap = torch.cat(dist_ap)
dist_an = torch.cat(dist_an)

# Compute loss
p = (dist_an-dist_ap-m+math.pi) / math.pi
p = p.clamp(min=eps, max=1-eps)
loss = -torch.log(p)

