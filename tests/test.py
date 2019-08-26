import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torch
from torch import nn
from network import MGN, Resnet, CGN, SN, FPN
import numpy as np
from PIL import Image
from torch.nn import functional as F

def get_layers(img):
    img = img.reshape(1, 3, 384, 128)
    layers = []
    for i in range(len(model.features)-1):
        img = model.features[i](img)
        layers.append(img)
    layers = [layer.squeeze() for layer in layers]
    return layers

def show_layers(img):
    layers = get_layers(img)

    img = i_transform(layers[0])
    img.save('pics/img_{}.jpg'.format(0))

    img = i_transform(layers[1].reshape(3, 512, 512))
    img.save('pics/img_{}.jpg'.format(1))

    img = i_transform(layers[2].reshape(3, 512, 512))
    img.save('pics/img_{}.jpg'.format(2))

    img = i_transform(layers[3].reshape(3, 512, 512))
    img.save('pics/img_{}.jpg'.format(3))

    img = i_transform(layers[4].reshape(3, 256, 256))
    img.save('pics/img_{}.jpg'.format(4))

    img = i_transform(layers[5].reshape(3, 512, 512))
    img.save('pics/img_{}.jpg'.format(5))

    img = i_transform(layers[6].reshape(3, 512, 256))
    img.save('pics/img_{}.jpg'.format(6))

    img = i_transform(layers[7].reshape(3, 256, 256))
    img.save('pics/img_{}.jpg'.format(7))

    img = i_transform(layers[8].reshape(3, 256, 128))
    img.save('pics/img_{}.jpg'.format(8))

def draw(channel_image, name):
    channel_image = channel_image.squeeze()
    w, h = channel_image.shape[0], channel_image.shape[1]
    t = np.zeros((w, h, 3))

    min_ = channel_image.max()
    max_ = channel_image.min()
    channel_image = (channel_image - min_) / (max_ - min_) * 255
    t[:,:, 0] = channel_image.detach().numpy()
    t[:,:, 1] = t[:,:,0]
    t[:,:, 2] = t[:,:,0]
    

    result = Image.fromarray(t.astype(np.uint8), mode='RGB')
    result.save('pics/'+name+'.png')

def spatial_attention(input1, input2):
    input1 = input1.unsqueeze(0)
    input2 = input2.unsqueeze(0)
    output = F.cosine_similarity(input1, input2, dim=0)
    return output

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

imgs = [origin_transforms(default_loader(path)) for path in paths]

model = Resnet()
checkpoint = torch.load('models/market1501/weights/Resnet/checkpoint_600.pth.tar')
model.load_state_dict(checkpoint['state_dict'], map_location='cuda')

###########################################

# img1 = get_layers(imgs[0])[1][10]
# img2 = get_layers(imgs[1])[1][10]
# img3 = get_layers(imgs[4])[1][10]

# output1_1 = spatial_attention(img1, img1)
# output1_2 = spatial_attention(img1, img2)
# output1_3 = spatial_attention(img1, img3)

# draw(img1, 'simg_1')
# draw(img2, 'simg_2')
# draw(img3, 'simg_3')
# draw(output1_1, 'out1_1')
# draw(output1_2, 'out1_2')
# draw(output1_3, 'out1_3')

############################################

layers1 = get_layers(imgs[4])
for i, layer in enumerate(layers1):
    print(layer.shape)
    n_features = layer.shape[0]
    n_rows = 4
    n_cols = 16
    img = torch.Tensor()
    for col in range(n_cols):
        img_row = torch.Tensor()
        for row in range(n_rows):
            channel_image = layer[(col*n_rows+row)]
            # channel_image -= channel_image.mean()
            # channel_image *= 64
            # channel_image = torch.clamp(channel_image, 0, 255)
            img_row = torch.cat([img_row, channel_image], dim=0)
        img = torch.cat([img, img_row], dim=1)
    draw(img, 'img_{}'.format(i))