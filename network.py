import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
from opt import opt

from torch.autograd import Variable
import torch.nn.functional as F

import random
import numpy as np

from torch.nn import Parameter
import math

# from PSP.load_psp import load_PSP
from util.utility import Show_Seg

# num_classes = 751  # change this depend on your dataset
num_classes = opt.num_classes

class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()

        feats = 256
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(self.reduction)

        self.fc_id_2048_0 = nn.Linear(feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, num_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x, labels=None):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        resnet = resnet50(pretrained=True)
        # self.base_params = nn.Sequential(*list(resnet.children())[:-3]).parameters()

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(2048, num_classes)
        self._initialize_fc(self.fc)

    def forward(self, x, labels=None):
        x = self.backbone(x) # torch.Size([8, 2048, 1, 1])
        feat = x.view(x.size(0), -1)
        prediction = self.fc(feat)
        return feat, prediction

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)

class CGN(nn.Module):
    def __init__(self):
        super(CGN, self).__init__()

        resnet = resnet50(pretrained=True)
        # self.base_params = nn.Sequential(*list(resnet.children())[:-3]).parameters()

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )


        self.conv1 = nn.Conv2d(2048//opt.Nc, opt.Cg, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tuple_fc = [0,]*opt.Nc
        self.tuple_bn = [0,]*opt.Nc
        for i in range(opt.Nc):
            self.tuple_fc[i] = nn.Linear(opt.Cg, num_classes)
            self.tuple_bn[i] = nn.BatchNorm2d(opt.Cg)
        self.tuple_fc = nn.ModuleList(self.tuple_fc)
        self.tuple_bn = nn.ModuleList(self.tuple_bn)
            
        # initialization    
        self._initialize_conv(self.conv1)
        for i in range(opt.Nc):
            self._initialize_fc(self.tuple_fc[i])
            self._initialize_norm(self.tuple_bn[i])

    def forward(self, x, labels=None):
        x = self.backbone(x) # torch.Size([8, 2048, 1, 1])
        feat = x.view(x.size(0), -1) # F: global feature
        tuple_x = torch.split(x, 2048//opt.Nc, 1) # f: channel groups torch.Size([8, 2048//8, 1, 1])

        predictions = [0,]*opt.Nc
        for i in range(opt.Nc):
            out = self.conv1(tuple_x[i]) #[8, 2048//8, 1, 1]
            out = self.tuple_bn[i](out)
            out = self.relu(out)
            out = out.view(x.size(0), -1)
            predictions[i] = self.tuple_fc[i](out)

        # torch.Size([8, 2048]) torch.Size([8, 751])
        # print(feat.shape, predictions[0].shape)
        return feat, predictions

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)
    
    def _initialize_conv(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
    
    def _initialize_norm(self, m):
        nn.init.normal_(m.weight, mean=1., std=0.02)
        nn.init.constant_(m.bias, 0.)

class SN(nn.Module):
    def __init__(self):
        super(SN, self).__init__()

        ## data generate
        batchsize = opt.batchimage *  opt.batchid
        assert opt.batchimage == 4 and batchsize == 16
        ix1 = np.array([0,   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        ix2 = np.array([14, 15, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 1, 8, 3, 9, 5, 2, 7, 3, 9, 4, 11,  5, 13,  6, 15, 7])
        VRandom = random.Random(opt.seed)
        tmp_ix1 = np.array(VRandom.choices(list(range(batchsize)), k=opt.cross_size))
        tmp_ix2 = np.array(VRandom.choices(list(range(batchsize)), k=opt.cross_size))
        self.base_ix1 = np.concatenate([ix1, tmp_ix1])
        self.base_ix2 = np.concatenate([ix2, tmp_ix2])
        self.softmax = nn.Softmax(dim=1)

        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(2048, num_classes)
        self._initialize_fc(self.fc)
            
    def forward(self, x):
        x = self.extract_feature(x)
        ckernal, cinput = x[self.ix1], x[self.ix2]
        coutput = self.attention(ckernal, cinput)
        coutput = self.classify(coutput)

        x = x.view(x.size(0), -1)
        prediction = self.fc(x)

        return prediction, coutput

    def extract_feature(self, x):
        x = self.features(x)
        return x

    def attention(self, ckernal, cinput):

        batch_size = ckernal.shape[0]

        cinput = cinput.view(batch_size, -1)
        ckernal = ckernal.view(batch_size, -1)

        coutput = F.cosine_similarity(ckernal, cinput, dim=1)

        # cinput = cinput.view(1, batch_size*256, 3, 3)
        # ckernal = ckernal.view(batch_size, 256, 3, 3)

        # cconv = nn.Conv2d(batch_size*256, batch_size, kernel_size=3, bias=False, groups=batch_size)
        # cconv.weight = nn.Parameter(ckernal)
        # coutput = cconv(cinput)
        # coutput = coutput.view(batch_size)

        return coutput

    def classify(self, coutput):
        # coutput = torch.sigmoid(coutput)
        coutput = coutput / 2. + 0.5
        coutput = torch.stack([1-coutput, coutput], dim=1)

        return coutput

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)

    def _init_conv(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    def generate_data(self, inputs, labels, cameras):
        valid = ((labels[self.base_ix1] != labels[self.base_ix2]) |
                 (cameras[self.base_ix1] != cameras[self.base_ix2]))
        valid = valid.cpu().numpy() == 1
        ix1, ix2 = self.base_ix1[valid], self.base_ix2[valid]
        ix1, ix2 = ix1[:opt.cross_size], ix2[:opt.cross_size]
        cross_labels = labels[ix1] == labels[ix2] # same 1, different 0
        cross_labels = cross_labels.long()
        labels = labels.to(opt.device)
        cross_labels = cross_labels.to(opt.device)
        self.ix1, self.ix2 = ix1, ix2
        return labels, cross_labels

class FPN(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=[3,4,6,3]): # FPN101  [2,4,23,3]
        super(FPN, self).__init__()

        feats = 768

        resnet = resnet50(pretrained=True)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        assert opt.batchimage == 4
        ix = np.arange(opt.cross_size)
        self.ix1 = ix
        self.ix2 = np.concatenate([ix[2:], ix[:2]])

        # Bottom-up layers
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 =  resnet.layer1
        self.layer2 =  resnet.layer2
        self.layer3 =  resnet.layer3
        self.layer4 =  resnet.layer4
        self.conv5 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.reduction = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # MAX POOLING
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(16, 16))
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(4, 4))
        self.maxpool_6 = nn.MaxPool2d(kernel_size=(2, 2))

        # classification
        self.fc_3 = nn.Linear(feats, 2)
        self.fc_4 = nn.Linear(feats, 2)
        self.fc_5 = nn.Linear(feats, 2)
        self.fc_6 = nn.Linear(feats, 2)
        self.fc_7 = nn.Linear(feats, 2)

        # init
        self._init_reduction(self.reduction)

        self._init_conv(self.conv5)
        self._init_conv(self.conv6)
        self._init_conv(self.toplayer)
        # self._init_conv(self.smooth1)
        # self._init_conv(self.smooth2)
        self._init_conv(self.latlayer1)
        self._init_conv(self.latlayer2)

        self._init_fc(self.fc_3)
        self._init_fc(self.fc_4)
        self._init_fc(self.fc_5)
        self._init_fc(self.fc_6)
        self._init_fc(self.fc_7)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners = False) + y

    def extract_feature(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv5(c5)
        p7 = self.conv6(self.relu(p6))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.reduction(p4)
        p3 = self.reduction(p3)

        feats = [0,]*5
        feats[0] = self.maxpool_3(p3)
        feats[1] = self.maxpool_4(p4)
        feats[2] = self.maxpool_5(p5)
        feats[3] = self.maxpool_6(p6)
        feats[4] = p7

        for i in range(5):
            feats[i] = feats[i].view(feats[i].size(0), -1)

        return feats

    def classify(self, feats, evaluate=True):
        p = [0,]*5

        p[0] = self.fc_3(feats[0])
        p[1] = self.fc_4(feats[1])
        p[2] = self.fc_5(feats[2])
        p[3] = self.fc_6(feats[3])
        p[4] = self.fc_7(feats[4])
        
        if evaluate:
            return self.eva(p)
        
        return p

    def eva(self, p):
        for i in range(5):
            p[i] = self.softmax(p[i])
        p = torch.stack(p, 0)
        p = p.mean(dim=0)
        return p

    def forward(self, x):
        feats = self.extract_feature(x)
        
        for i in range(5): 
            feats[i] = feats[i][self.ix1] - feats[i][self.ix2]
        
        p = self.classify(feats, evaluate=False)

        return p

    def generate_data(self, inputs, labels):
        cross_labels = labels[self.ix1] != labels[self.ix2] # same 1, different 0
        cross_labels = cross_labels.long()
        return cross_labels

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    def _init_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)
    
    def _init_conv(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')

class AN(nn.Module):
    def __init__(self):
        super(AN, self).__init__()

        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(2048, num_classes)
        self._initialize_fc(self.fc)

        # self.metric_fc = ArcMarginProduct(2048, num_classes, s=30, m=0.5)

    def forward(self, x, labels):
        x = self.backbone(x)
        feats = x.view(x.size(0), -1)

        output = self.fc(feats)
        # output = self.metric_fc(feats, labels)

        return feats, output

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)
        
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, eps=1e-5):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.eps = eps
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(min=-1+self.eps, max=1-self.eps)
        theta = torch.acos(cosine)
        phi = theta + self.m
        theta = torch.clamp(math.pi - theta, self.eps, math.pi-self.eps)
        phi = torch.clamp(math.pi - phi, self.eps, math.pi-self.eps)
        
        # sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # if self.easy_margin:
        #     phi = torch.where(cosine > 0, phi, cosine)
        # else:
        #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=opt.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * theta) 
        output *= self.s

        return output

class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()

        self.num_branches = len(opt.branches)
        self.draw = Show_Seg()
        feat = opt.feat
        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )

        feature = nn.Sequential(
            resnet.layer3,
            resnet.layer4
        )

        self.reduction = nn.Sequential(nn.Conv2d(2048, feat, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self._init_reduction(self.reduction)

        self.tuple_backbone = [0,]* self.num_branches
        for i in range(self.num_branches):
            self.tuple_backbone[i] = copy.deepcopy(feature)
        self.tuple_backbone = nn.ModuleList(self.tuple_backbone)
        
        self.fc2 = nn.Linear(feat*self.num_branches, num_classes)
        self._initialize_fc(self.fc2)

        # self.glo_f = nn.Sequential(copy.deepcopy(feature), nn.AdaptiveAvgPool2d((1, 1)))
        # self.fc = nn.Linear(2048, num_classes)
        # self._initialize_fc(self.fc)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def seg_seg(bi_img):
        raw_sum = bi_img.sum(dim=-1)  #384
        col_sum = bi_img.sum(dim=-2)  #128
        j1, j2, j3, j4 = -1, -1, -1, -1
        r, c = raw_sum.shape[0], col_sum.shape[0]
        for j in range(r):
            if raw_sum[j] > 0:
                j1 = j
                break
        for j in range(r-1, -1, -1):
            if raw_sum[j] > 0:
                j2 = j
                break
        for j in range(c):
            if col_sum[j] > 0:
                j3 = j
                break
        for j in range(c-1, -1, -1):
            if col_sum[j] > 0:
                j4 = j
                break
        return j1, j2, j3, j4

    @staticmethod
    def expand(bi_img, img, c, h, w):
        img = img * bi_img.float()

        seg_img = img[:, c[0]:c[1]+1, c[2]:c[3]+1]
        seg_img = seg_img.unsqueeze(0)
        seg_img = F.interpolate(seg_img, size=(h, w), mode='bilinear', align_corners=False)
        seg_img = seg_img.squeeze(0)
        return seg_img

    def forward(self, x, pred_segs=None, labels=None):
        n = x.shape[0] # 16, 3, 256, 256
        h, w = 384, 128
        no_img = torch.FloatTensor(3, h, w).zero_().to(opt.device)

        # global_f = self.backbone(x) # torch.Size([8, 2048, 1, 1])
        # global_f = self.glo_f(global_f)
        # global_f = global_f.view(global_f.size(0), -1)
        # global_p = self.fc(global_f)

        part_f = torch.FloatTensor().to(opt.device)
        features = [0] * self.num_branches
        part_f = [0] * n

        # self.draw.show_pred_seg('./tmp', pred_segs[0], '')
        # self.draw.show('./tmp', x[0], '')

        for j in range(self.num_branches):
            _b = opt.branches[j]  
            bi_imgs = (pred_segs == _b[0])     # 16, 256, 256
            if len(_b) > 1:
                for _ in _b: 
                    bi_imgs = bi_imgs | (pred_segs == _)    
            imgs = [0] * n  
            for i in range(n):
                bi_img = bi_imgs[i] # 384, 128
                if bi_img.sum() > 0:
                    c = self.seg_seg(bi_img)
                    img = self.expand(bi_img, x[i], c, h, w)
                    imgs[i] = img
                else:
                    imgs[i] = no_img
            imgs = torch.stack(imgs, dim=0)
            
            feature = self.backbone(imgs) #
            feature = self.tuple_backbone[j](feature)
            feature = self.reduction(feature)
            feature = feature.squeeze(3).squeeze(2) # 16, 128
            features[j] = feature

            # self.draw.show('./tmp', imgs[0], j)

        del imgs, pred_segs, x

        for i in range(n):
            part_f[i] = torch.cat([features[j][i] for j in range(self.num_branches)], dim=0) # 20, 128 

        part_f = torch.stack(part_f, dim=0) # 16, 2560

        part_p = self.fc2(part_f)

        # return global_f, part_f, global_p, part_p
        return part_f, part_p

    def _initialize_fc(self, m):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.)