from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from util.RandomErasing import RandomErasing
from util.RandomSampler import RandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from opt import opt
import os
import re
import random
import json
from pathlib import Path
import torch

def stratify_sample(labels, seed, val_size):
    X = np.arange(len(labels))
    y = np.asarray(labels)
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=val_size,
                                                    random_state=seed)
    return X_train, X_val

class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((opt.h, opt.w), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((opt.h, opt.w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if "viva" in opt.data_path:
            pre_viva = Vivalab_pre()
            # pre_viva.check_tids(opt.data_path)
            train_imgs, val_imgs, test_imgs, query_imgs, self.tid_dict, self.test_query_imgs = pre_viva.get_all(opt.data_path)
            self.test_transform = test_transform
            self.trainset = Vivalab(train_transform, train_imgs, 'train')
            self.testset = Vivalab(test_transform, test_imgs, 'test')
            self.queryset = Vivalab(test_transform, query_imgs, 'query')
            self.valset = Vivalab(test_transform, val_imgs, 'val')
        else:
            self.trainset = Market1501(train_transform, 'train', opt.data_path)
            self.testset = Market1501(test_transform, 'test', opt.data_path)
            self.queryset = Market1501(test_transform, 'query', opt.data_path)
            self.valset = Market1501(test_transform, 'val', opt.data_path)

        pin_memory = opt.device != 'cpu'
        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=opt.batchid,
                                                                        batch_image=opt.batchimage),
                                                  batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers,
                                                  drop_last = True,
                                                  pin_memory = pin_memory)
        self.val_loader = dataloader.DataLoader(self.valset, 
                                                sampler=RandomSampler(self.valset, batch_id=opt.batchid,
                                                            batch_image=opt.batchimage),
                                                batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers,
                                                drop_last = True,
                                                pin_memory = pin_memory)
        self.test_loader = dataloader.DataLoader(self.testset, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory = pin_memory)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=opt.batchquery, num_workers=opt.num_workers, pin_memory = pin_memory)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))

class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.to_tensor = transforms.ToTensor()
        self.to_img = transforms.ToPILImage()

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train' or dtype == 'val':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        if opt.debug:
            self.imgs = self.imgs[:128]

        img_ids = [self.id(path) for path in self.imgs]

        if dtype in ['train', 'val']:
            train_index, val_index = stratify_sample(img_ids, opt.seed, 0.1)
            if dtype == 'train':
                self.imgs = [self.imgs[i] for i in train_index]  # 98797
            else:
                self.imgs = [self.imgs[i] for i in val_index] # 10978
        
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]

        path_npz = path[:-3] + 'npz'
        seg = np.load(path_npz)['data']
        seg = torch.from_numpy(seg)

        target = self._id2label[self.id(path)]
        
        # cam = self.camera(path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, seg

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])
    
    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

class Vivalab(dataset.Dataset):
    def __init__(self, transform, imgs, dtype=''):
        self.transform = transform
        if opt.trans == 0:
            self.loader = self.img_loader
        else:
            self.loader = default_loader
        
        self.imgs = imgs               
        
        if dtype != '': print(dtype, len(self.imgs))

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

        self.cameras = [self.camera(path) for path in self.imgs]

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]
        img = self.loader(path)
        cam = self.camera(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, cam

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def img_loader(file_path):
        with open(file_path, 'rb') as f:
            _ = Image.open(f)
            _.load()
            image = Image.new("RGB", _.size, "WHITE")
            image.paste(_, (0, 0), _)
        _ = None
        del _
        return image

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        p = file_path.with_suffix('.json')
        with p.open() as f:
            temp = json.loads(f.read())
            c = int(temp['CAMERA'])
        return c

    @staticmethod
    def tid(file_path):
        """
        :param file_path: unix style file path
        :return: tracklet id
        """
        return int(file_path.parts[-2])

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """

        return int(file_path.parts[-3])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    def list_pictures(self, path):
        paths = Path(path).glob('**/*.png')
        return paths

class Vivalab_pre():
    def __init__(self):
        pass

    @staticmethod
    def list_pictures(path):
        paths = Path(path).glob('**/*.png')
        return paths   

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.parts[-3])

    @staticmethod
    def tid(file_path):
        """
        :param file_path: unix style file path
        :return: tracklet id
        """
        return int(file_path.parts[-2])

    @staticmethod
    def check_tids(datadir):
        data_path = datadir
        all_ids = os.listdir(data_path)     
        all_tids = []
        for _ in all_ids:
            all_tids += os.listdir(data_path+'/'+_)
        assert len(all_tids) == len(set(all_tids))

    def get_all(self, datadir):
        data_path = datadir
        all_ids = os.listdir(data_path)
        all_ids = [int(_) for _ in all_ids]
        imgs = [path for path in self.list_pictures(data_path)]
        
        VRandom = random.Random(opt.seed)
        VRandom.shuffle(all_ids)
        VRandom.shuffle(imgs)

        assert all_ids[:14] == [34073, 34530, 34368, 34088, 33966, 34382, 34475, 33897, 34362, 34061, 34437, 34254, 34166, 34089]

        _len = len(all_ids)
        assert _len // 2 == opt.num_classes
        print("num of categories", _len)

        ## train & val
        train_val_ids = all_ids[:_len//2]
        train_val_imgs = [path for path in imgs if self.id(path) != -1 and self.id(path) in train_val_ids]
        labels = [self.id(path) for path in train_val_imgs]
        train_index, val_index = stratify_sample(labels, opt.seed, 0.1)
        train_imgs = [train_val_imgs[i] for i in train_index]  # 98797
        val_imgs = [train_val_imgs[i] for i in val_index] # 10978

        ## test query
        test_query_ids = all_ids[_len//2:]
        if opt.debug:
            test_query_ids = test_query_ids[:2]
        test_query_imgs = [path for path in imgs if self.id(path) != -1 and self.id(path) in test_query_ids]

        ## id dict
        id_dict = {}
        for path in test_query_imgs:
            path_id = self.id(path) 
            if path_id in id_dict:
                if len(id_dict[path_id]) < 80:
                    id_dict[path_id].append(path)
            else:
                id_dict[path_id] = [path,]
     
        ## test & query
        test_imgs, query_imgs  = [], []
        # tmp = [len(y) for y in list(id_dict.values())]
        # print(max(tmp), min(tmp)) # 20 5733
        for path_id in list(id_dict.keys()):
            paths = id_dict[path_id]
            # VRandom.shuffle(paths) have been shuffled before for the whole dataset
            _n = round(len(paths) * 0.1419)
            query_imgs += paths[:_n] # 2166
            test_imgs += paths[_n:]  # 13555        

        ## tid_dict
        test_query_tids = np.array([self.tid(path) for path in test_query_imgs])
        arg_ = np.argsort(test_query_tids)
        test_query_imgs = np.array(test_query_imgs)
        test_query_imgs = test_query_imgs[arg_]
        test_query_imgs = test_query_imgs.tolist()
        del arg_, test_query_tids
        tid_dict = {}
        for i in range(len(test_query_imgs)):
            path = test_query_imgs[i]
            path_tid = self.tid(path) 
            if path_tid not in tid_dict:
                tid_dict[path_tid] = i

        return train_imgs, val_imgs, test_imgs, query_imgs, tid_dict, test_query_imgs