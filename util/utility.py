import matplotlib
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np

def cut_range(n, batch):
    cuts = n // batch
    cuts = [i*batch for i in range(cuts)]
    cuts.append(n)
    return cuts

class Show_Seg(object):
    def __init__(self):
        self.classes = np.array(('Background',  # always index 0
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
        self.cmap = matplotlib.colors.ListedColormap(colormap)
        bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.norm = matplotlib.colors.BoundaryNorm(bounds, self.cmap.N)
        self.to_img = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    def show_pred_seg(self, savedir, pred_seg, i):
        """
        pred_seg: Tensor
        """
        plt.imshow(pred_seg.cpu().numpy(), cmap=self.cmap, norm=self.norm)
        plt.savefig(fname=savedir+"/seg_{}.png".format(i))
        plt.close()
    
    def show(self, savedir, img, i):
        """
        img: Tensor
        """
        img = self.to_img(img.cpu())
        img.save(savedir+"/img_{}.png".format(i))