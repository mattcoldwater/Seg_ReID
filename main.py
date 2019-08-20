from opt import opt
import os
if opt.device != 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

import json
from sklearn.metrics import auc
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from torch import multiprocessing as mp

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import dataloader

from torchvision import transforms
from torchvision.datasets.folder import default_loader

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from data import Data, Vivalab
from util.extract_feature import extract_feature, extract_feature_SN
from util.get_optimizer import get_optimizer
from util.metrics import mean_ap, cmc, re_ranking, mean_ap_, cmc_
from util.utility import cut_range
from network import MGN, Resnet, CGN, SN, FPN, AN, Segnet
from loss import Loss_MGN, Loss_Resnet, Loss_CGN, Loss_SN, Loss_FPN, Loss_AN, Loss_Segnet

class Main():
    
    def __init__(self, model, loss, data=None):
        if data != None:
            if 'viva' in opt.data_path:
                self.test_transform = data.test_transform
                self.tid_dict = data.tid_dict
                self.test_query_imgs = data.test_query_imgs
            self.train_loader = data.train_loader
            self.test_loader = data.test_loader
            self.val_loader = data.val_loader
            self.query_loader = data.query_loader
            self.testset = data.testset
            self.queryset = data.queryset

        self.nGPU = torch.cuda.device_count()
        self.model = model

        if self.nGPU > 1 and opt.device != 'cpu':
            self.model = nn.DataParallel(self.model, device_ids=range(self.nGPU))

        self.model = self.model.to(opt.device)
        self.loss = loss
        self.optimizer = get_optimizer(self.model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.model.train()

        for batch, (inputs, labels, segs) in enumerate(self.train_loader):
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)
            segs = segs.to(opt.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, segs, labels)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()

    def val(self):

        self.model.eval()
        torch.cuda.empty_cache()

        losses = []
        with torch.no_grad():
            for batch, (inputs, labels, segs) in enumerate(self.val_loader):
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)
                segs = segs.to(opt.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs, segs, labels)
                loss = self.loss(outputs, labels, False)
                losses.append(loss.cpu().numpy())
        
        print('Val Loss {:.4f}'.format(sum(losses)/len(losses)))
        torch.cuda.empty_cache()

    def evaluate(self):

        if opt.debug:
            return 0

        if opt.model_name in opt.SN_family:
            self.evaluate_SN()
            return 0

        self.topk = 20
        self.model.eval()
        torch.cuda.empty_cache()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def eff_rank(qf, gf, batch_size=10):

            cuts = cut_range(qf.shape[0], batch_size)
            # Compute CMC for each query
            ret = np.zeros(self.topk)
            num_valid_queries = 0
            # Compute AP for each query
            aps = []

            for _ in range(len(cuts)-1):
                i, j = cuts[_], cuts[_+1]
                batch_mat = cdist(qf[i:j, :], gf)
                ret, num_valid_queries = cmc(batch_mat, ret, num_valid_queries, 
                    self.queryset.ids[i:j], self.testset.ids, self.queryset.cameras[i:j], self.testset.cameras, self.topk,
                    separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
                aps = mean_ap(batch_mat, aps, 
                    self.queryset.ids[i:j], self.testset.ids, self.queryset.cameras[i:j], self.testset.cameras)
            return ret.cumsum() / num_valid_queries, np.mean(aps)

        def rank(dist):
            r = cmc_(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras, self.topk,
                    separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
            m_ap = mean_ap_(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        if opt.re_rank:
            #########################   re rank##########################
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

            r, m_ap = rank(dist)

            print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
                  .format(m_ap, r[0], r[2], r[4], r[9]))
            #########################no re rank##########################
        
        # dist = cdist(qf, gf)
        # r, m_ap = rank(dist)
    
        r, m_ap = eff_rank(qf, gf)
        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
              
        torch.cuda.empty_cache()

    def evaluate_SN(self):    
        self.topk = 20
        self.model.eval()
        torch.cuda.empty_cache() 

        with torch.no_grad():
            qf = extract_feature_SN(self.model, tqdm(self.query_loader)) # 3340, 2048
            gf = extract_feature_SN(self.model, tqdm(self.test_loader))
            torch.cuda.empty_cache()        

        def eff_rank_SN(qf, gf, batch_size=512):

            with torch.no_grad():

                if opt.model_name == 'FPN':
                    gf_len = gf[0].shape[0]
                    qf_len = qf[0].shape[0]
                else:
                    gf_len = gf.shape[0]
                    qf_len = qf.shape[0]
                cuts = cut_range(gf_len, batch_size)
                # Compute CMC for each query
                ret = np.zeros(self.topk)
                num_valid_queries = 0
                # Compute AP for each query
                aps = []

                for k in tqdm(range(qf_len)):
                    batch_mat = [0]*(len(cuts)-1)
                    for _ in range(len(cuts)-1):
                        i, j = cuts[_], cuts[_+1]
                        if opt.model_name == 'FPN':
                            inputs = [0,]*5
                            for r in range(5):
                                inputs[r] = torch.stack([qf[r][k,:], ]*(j-i), 0) - gf[r][i:j, :]
                                inputs[r].to(opt.device)
                        else:
                            inputs1 = torch.stack([qf[k], ]*(j-i), 0)
                            inputs2 = gf[i:j]
                            inputs1.to(opt.device)
                            inputs2.to(opt.device)
                            inputs = self.model.attention(inputs1, inputs2)
                        outputs = self.model.classify(inputs)
                        outputs = outputs.to('cpu').numpy()
                        outputs = outputs[:, 0]
                        batch_mat[_] = outputs
                    batch_mat = np.concatenate(batch_mat)
                    batch_mat = batch_mat.reshape(1, -1)
                    ret, num_valid_queries = cmc(batch_mat, ret, num_valid_queries, 
                        self.queryset.ids[k:k+1], self.testset.ids, self.queryset.cameras[k:k+1], self.testset.cameras, self.topk,
                        separate_camera_set=False, single_gallery_shot=False, first_match_break=True)
                    aps = mean_ap(batch_mat, aps, 
                        self.queryset.ids[k:k+1], self.testset.ids, self.queryset.cameras[k:k+1], self.testset.cameras)

                return ret.cumsum() / num_valid_queries, np.mean(aps)
        
        r, m_ap = eff_rank_SN(qf, gf)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
              
        torch.cuda.empty_cache()

    def simi(self):
        torch.cuda.empty_cache()

        test_transforms  = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if opt.file1 != '' and opt.file2 != '':
            paths = [opt.file1, opt.file2]
        else:
            paths = list(Path(opt.data_path).glob('**/*.png'))
            names = [path.parts[-1] for path in paths]
        
        imgs = [test_transforms(default_loader(path)) for path in paths]
        inputs = torch.stack(imgs, dim=0)
        print('Whole Imgs Tensor', inputs.shape)

        features = torch.FloatTensor()
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to(opt.device)
            outputs = model(input_img)

            f = outputs[0].data.cpu()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0).numpy()

        dist = cdist(features, features)
        if len(imgs) == 2:
            print(dist[0, 1])
        else:
            dist = pd.DataFrame(data=dist, index=names, columns=names)
            print(dist)

    def roc(self):

        os.makedirs('logs/', exist_ok=True)

        self.model.eval()

        ## feature extraction
        torch.cuda.empty_cache()
        loader = dataloader.DataLoader(
            Vivalab(self.test_transform, self.test_query_imgs), batch_size=16, num_workers=opt.num_workers, pin_memory = opt.device!='cpu')
        f = extract_feature(self.model, tqdm(loader)).numpy()
        print('num of images:', f.shape[0]) #, len(self.test_query_imgs)
        torch.cuda.empty_cache()
        del loader
        
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.query_loader = None
        self.testset = None
        self.queryset = None
        self.model = None

        ## dist & best match
        tids = list(self.tid_dict.keys())
        _len = len(tids)
        tid_tuple = sorted(self.tid_dict.items(), key=lambda x: x[1])
        tid_tuple.append((-1, len(self.test_query_imgs)))
        results, labels = np.zeros(_len*(_len-1)//2), np.zeros(_len*(_len-1)//2)
        _n = 0
        for i in tqdm(range(_len)):

            t1, s1, e1 = tid_tuple[i][0], tid_tuple[i][1], tid_tuple[i+1][1]
            id1 = int(self.test_query_imgs[s1].parts[-3])
            f1 = f[s1:e1, :]
            dist = cdist(f1, f)
            del f1
            # print(dist.shape)

            for j in range(i+1, _len):

                t2, s2, e2 = tid_tuple[j][0], tid_tuple[j][1], tid_tuple[j+1][1]
                id2 = int(self.test_query_imgs[s2].parts[-3])
                mat = dist[:, s2:e2]
                min_dist = mat.min()
                del mat
                results[_n] = min_dist
                labels[_n] = int(id1 == id2)
                _n += 1

        ## accuracy curve
        thresholds = results.copy()
        thresholds.sort()
        tps, tns = np.zeros(_len*(_len-1)//2), np.zeros(_len*(_len-1)//2)
        condition_pos = np.sum(labels == 1) # 341
        condition_neg = np.sum(labels == 0) # 91465
        for _ in range(len(thresholds)):
            th = thresholds[_] + 1e-8
            tps[_] = np.sum((results <= th) & (labels == 1)) / condition_pos * 100
            tns[_] = np.sum((results > th) & (labels == 0)) / condition_neg * 100
        
        roc_dict = {
            'ths': thresholds.tolist(),
            'tps': tps.tolist(),
            'tns': tns.tolist(),
        }

        ## draw TP TN
        plt.plot(thresholds, tps, label='TP')
        plt.plot(thresholds, tns, label="TN")
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('%')
        plt.yticks(np.arange(0, 100, 5))
        plt.grid(True)
        plt.savefig('logs/TPTN_{}_{}.jpg'.format(opt.model_name, opt.trans))
        plt.close()
       
        ## roc
        tps = tps / 100
        fps = 1 - tns/100
        arg_ = np.argsort(fps)
        fps = fps[arg_]
        tps = tps[arg_]
        auc_ = auc(fps, tps)
        plt.plot(fps, tps)
        plt.xticks(np.arange(0, 1, 0.1))
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.grid(True)
        plt.savefig('logs/ROC_{}_{}.jpg'.format(opt.model_name, opt.trans))
        plt.close()

        ## json 
        roc_dict['auc'] = auc_
        with open('logs/roc_{}_{}.json'.format(opt.model_name, opt.trans), 'w', encoding='utf-8') as f:
            json.dump(roc_dict, f)

    def save(self, epoch, filename):
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, filename)

    def load(self, filename):
        filename = opt.weight_path + '/checkpoint_{}.pth.tar'.format(filename)
        checkpoint = torch.load(filename, map_location=torch.device(opt.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1, last_epoch=checkpoint['epoch'])
        print("Loading from epoch:", checkpoint['epoch'], 'schedular:', self.scheduler.last_epoch)
        return checkpoint['epoch']

if __name__ == '__main__':  
    mp.set_start_method(opt.start_method, True)   
    
    if opt.model_name == 'MGN':
        model = MGN()
        loss = Loss_MGN()
    elif opt.model_name == 'Resnet':
        model = Resnet()
        loss = Loss_Resnet()
    elif opt.model_name == 'CGN':
        model = CGN()
        loss = Loss_CGN()
    elif opt.model_name == 'SN':
        model = SN()
        loss = Loss_SN()
    elif opt.model_name == 'FPN':
        model = FPN()
        loss = Loss_FPN()   
    elif opt.model_name == 'AN':
        model = AN()
        loss = Loss_AN()     
    elif opt.model_name == 'Segnet':
        model = Segnet()
        loss = Loss_Segnet()
        
    if opt.mode == 'train':
        main = Main(model, loss, Data())

        init_epoch = main.load(opt.weight) if opt.weight != -1 else 0
        os.makedirs(opt.weight_path, exist_ok=True)

        for epoch in range(init_epoch + 1, opt.epoch + 1):
            print('\nepoch', epoch)
            main.train()
            if epoch % 100 == 0 or epoch==1:
                print('\nstart evaluate')
                main.val()
                main.evaluate()
                main.save(epoch, opt.weight_path+'/checkpoint_{}.pth.tar'.format(epoch))

    if opt.mode == 'evaluate':
        main = Main(model, loss, Data())
        print('start evaluate')
        if opt.weight != -1:
            main.load(opt.weight)
        main.val()
        main.evaluate()

    if opt.mode == 'roc':
        main = Main(model, loss, Data())
        print('ROC curve')
        main.load(opt.weight)
        main.roc()

    if opt.mode == 'simi':
        main = Main(model, loss)
        print('Similarity scores')
        main.load(opt.weight)
        main.simi()