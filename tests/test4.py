import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from network import MGN, Resnet, CGN, SN, FPN, AN

from PIL import Image

import torch
from torch import nn

from torch.autograd import Variable
from torch.optim import Adam
from torch.nn import functional as F
from torch.distributions.normal import Normal

from PSP.load_psp import load_PSP



torch.manual_seed(100)

paths = [
    '0002_c1s1_000451_03.jpg',
    '0002_c1s1_000776_01.jpg',
    '0007_c3s3_077419_03.jpg',
    '0007_c2s3_070952_01.jpg',
    '0010_c6s4_002427_02.jpg',
    '0010_c6s4_002452_02.jpg']
paths = ['../market1501/Market1501/bounding_box_train/'+_ for _ in paths]

inter_trans_img = transforms.Resize((256*3, 256*3), interpolation=3)

to_tensor = transforms.ToTensor()
to_img = transforms.ToPILImage()

img = Image.open(paths[1])
img = inter_trans_img(img)
ten = to_tensor(img)
ten = ten.unsqueeze(0)
with torch.no_grad():
    model = load_PSP(device='cpu')
    model.eval()
    pred_seg, pred_cls = model(ten.to('cpu'))
    pred_seg = pred_seg[0].argmax(dim=0)

img.save('img.png')

img_1 = img.convert('L')
img_1.save('img1.png')

tensor = to_tensor(img_1)
img_2 = to_img(tensor)
img_2.save('img2.png')

tensor1 = to_tensor(img)
tensor1 = torch.stack([tensor1[0], tensor1[0], tensor1[0]])
img_3 = to_img(tensor1)
img_3.save('img3.png')