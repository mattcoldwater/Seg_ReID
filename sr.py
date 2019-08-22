from torchvision import transforms
from torchvision.datasets.folder import default_loader

path = 'test_samples/img_005'

sr = transforms.Resize((512, 256))

img = default_loader(path +'.jpg')
img = sr(img)
img.save(path + '_bi.jpg')