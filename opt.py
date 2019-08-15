import argparse

parser = argparse.ArgumentParser(description='reid')

# !python main.py --mode train --device cuda --weight_path models/viva/weights/Segnet --model_name Segnet --data_path ../market1501 --num_classes 751 --gpu 0 --start_method fork
# !python main.py --weight_path models/market1501/weights/Resnet --model_name Resnet --mode train --num_workers 16 --device cuda --data_path ../market1501 --num_classes 751 --gpu 0 --start_method fork
# python main.py --weight_path models/market1501/weights/Resnet --model_name Resnet --mode train --num_workers 16 --device cpu --data_path ../market1501/Market1501 --num_classes 751 --debug True

## global
parser.add_argument('--weight_path', default="models/viva/weights/bb_CGN", help='pytorch model checkpoint path')
parser.add_argument("--model_name", default='CGN', choices=['CGN', 'Resnet', 'MGN', 'AN', 'SN', 'FPN', 'Segnet'], help='what model you choose')
parser.add_argument('--mode', default='train', choices=['train', 'evaluate', 'roc', 'simi'], help='running mode')
parser.add_argument('--weight', default=-1, type=int, help='iteration number of checkpoint that is resumed from, -1 means training from start')
parser.add_argument("--num_workers", default=0, type=int, help='CPU multiprocessing for IO, 0 means no multiprocessing')
parser.add_argument("--device", default='cuda', help = 'cuda or cuda:0 or cpu')
parser.add_argument('--debug', default=False, type=bool, help='debug mode, smaller dataset')
parser.add_argument('--gpu', default='1', help='the gpu you use')
parser.add_argument('--start_method', default='spawn', help='mutiprocessing start method')
parser.add_argument('--branches', default=[5,], help='seg branches') #2, 5, 9, 13, 14, 15, 16, 17
""" 
'Background',  'Hat',          'Hair',      'Glove',     
'Sunglasses',  'UpperClothes', 'Dress',     'Coat',      
'Socks', 'Pants',        'Jumpsuits', 'Scarf',     'Skirt',
'Face',         'Left-arm',  'Right-arm', 'Left-leg',
'Right-leg',    'Left-shoe', 'Right-shoe',
"""
## data
parser.add_argument("--seed", default=100, type=int, help='random seed for data preparation')
parser.add_argument("--trans", default=1, type=int, help='transparency: 0 mask, 1 bounding box')
parser.add_argument('--data_path', default="../viva_dataset/viva_dataset", help='data path, Eg: ../market1501/Market1501')   

## training 
parser.add_argument("--num_classes", default=204, type=int, help='number of classes for training, market1501: 751, vivalab 204')
parser.add_argument('--epoch', default=500, type=int, help='total number of epoches to train')
parser.add_argument('--lr_scheduler', default=[320, 380], help='MultiStepLR, decay the learning rate')
parser.add_argument('--lr', default=2e-4, help='initial learning_rate')
parser.add_argument("--batchimage", default=4, type=int, help='images per id in a training batch')
parser.add_argument("--batchid", default=4, type=int, help='ids in a training batch')
parser.add_argument("--Nc", default=8, type=int, help='only works for CGN, in the paper')
parser.add_argument("--Cg", default=256, type=int, help='only works for CGN, in the paper')

## evaluation
parser.add_argument("--batchtest", type=int, default=16, help='the batch size for test')
parser.add_argument("--batchquery", type=int, default=16, help='the batch size for query')
parser.add_argument('--re_rank', default=False, help='re ranking technique')

## similarity
parser.add_argument('--file1', default='', help='only works when mode is simi, '': means more than 2 images, Eg: test_samples/114486.png')
parser.add_argument('--file2', default='', help='only works when mode is simi, '': means more than 2 images, Eg: test_samples/302389.png') 

## depreciated
parser.add_argument('--SN_family', default=['SN', 'FPN'], help='Siamese Network Family(depreciated)')
parser.add_argument('--query_image', default='0001_c1s1_001051_00.jpg', help='path to the image you want to query (depreciated)')
parser.add_argument('--freeze', default=False, help='freeze backbone or not(depreciated)')

opt = parser.parse_args()
