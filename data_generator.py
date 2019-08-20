from PSP.load_psp import load_PSP
import torch
import os
from pathlib import Path
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    paths =  [path for path in Path('/home/haoyu/re_id/datasets/Market1501').glob('**/*.jpg')]

    trans = transforms.Compose([
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    psp = load_PSP(device='cuda')
    psp.eval()
    with torch.no_grad():
        for path in tqdm(paths):
            path_npz = path.with_suffix('.npz')
            img = default_loader(path)
            img = trans(img)
            img = img.unsqueeze(0)
            img = img.to('cuda')
            pred_segs, pred_cls = psp(img)
            pred_segs = pred_segs.argmax(dim=1) # 16, 256, 256
            data = pred_segs[0].cpu().numpy()
            np.savez_compressed(path_npz, data=data)