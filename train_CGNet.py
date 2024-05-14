#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
from torch import optim
from utils import data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.utils import clip_gradient, adjust_lr
from utils.metrics import Evaluator

from network.CGNet import HCGMNet,CGNet

import time
start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, val_loader, Eva_train, Eva_val, save_path, net, criterion, optimizer, num_epoches):
    global best_iou
    epoch_loss = 0
    net.train(True)

    length = 0
    st = time.time()
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A,B)
        loss = criterion(preds[0], Y)  + criterion(preds[1], Y)
        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)
        
        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU, Pre, Recall, F1))
    print("Strat validing!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)
            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))


if __name__ == '__main__':
    seed_everything(42)

    train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
    val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
    batchsize = 8
    trainsize = 256
    lr = 5e-4
    
    epoch = 50

    train_loader = data_loader.get_loader(train_root, batchsize, trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(val_root, batchsize, trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)
    
    save_path = './output/'

    model = CGNet().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_iou = 0.0

    print("Start train...")


    for epoch in range(1, epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, save_path, model, criterion, optimizer, epoch)
        lr_scheduler.step()