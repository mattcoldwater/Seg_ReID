import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from network import MGN, Resnet, CGN, SN, FPN, AN

import torch
from torch import nn

from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions.normal import Normal

def draw(channel_image, name):
    channel_image = channel_image.squeeze()

    # t1 = torch.Tensor([128]*384*128).reshape(384, 128)
    # t2 = torch.Tensor([0]*384*128).reshape(384, 128)
    # channel_image = inter_transform(channel_image)
    # channel_image = torch.stack([channel_image[0], t1, t2], dim=0)

    channel_image = i_transform(channel_image)
    channel_image.save('testing/simg_{}.png'.format(name))

def get_layers(img):
    img = img.reshape(1, 3, 384, 128)
    layers = []
    for i in range(len(model.features)-1):
        img = model.features[i](img)
        layers.append(img)
    layers = [layer.squeeze() for layer in layers]
    return layers

def attention(input1, input2):
    input1 = input1.unsqueeze(0)
    input2 = input2.unsqueeze(0)
    output = F.cosine_similarity(input1, input2, dim=0)
    # output = input1 * input2
    # print(output.sum())
    return output
    
torch.manual_seed(100)

paths = [
    '0002_c1s1_000451_03.jpg',
    '0002_c1s1_000776_01.jpg',
    '0007_c3s3_077419_03.jpg',
    '0007_c2s3_070952_01.jpg',
    '0010_c6s4_002427_02.jpg',
    '0010_c6s4_002452_02.jpg']
paths = ['../market1501/Market1501/bounding_box_train/'+_ for _ in paths]

origin_transforms  = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inter_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 128), interpolation=3),
    transforms.ToTensor(),
])

i_transform = transforms.Compose([
    transforms.ToPILImage()
])

imgs = [origin_transforms(default_loader(path)) for path in paths]

model = AN()
checkpoint = torch.load('market1501/weights/AN/checkpoint_100.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

x = model.state_dict()
x1 = x['metric_fc.weight']

# model = Resnet()
# checkpoint = torch.load('market1501/weights/Resnet/checkpoint_600.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])

# x = model.state_dict()
# x2 = x['fc.weight'].norm(dim=0)

print(x1[0].shape)

