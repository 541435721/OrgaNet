#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_with_all_data.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/12/4 8:54
# @Desc  :

import os
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import gc
import numpy as np
from Models import BaseModel, MultiHeadClassifier, MeasureModel, Resnet, VggNet, InceptionNet, GoogleNet
from Data import Organoid_classification, NoiseValOrganoid, NoiseSharedOrganoid, NoisePrivateOrganoid, \
    NoiseConsensusOrganoid, NoiseOrganoid
from Loss import MAELoss, GCELoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import tensorboardX
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
import cv2
from itertools import chain


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser('train_params')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--epoch', type=int, default=10,
                    help='Epochs')
parser.add_argument('--batchsize', type=int, default=20,
                    help='Batch size')
parser.add_argument('--pretrain', type=int, default=1,
                    help='whether pretrain')
parser.add_argument('--hybird', type=int, default=0,
                    help='mutil input')
parser.add_argument('--worker', type=int, default=0,
                    help='worker id')
parser.add_argument('--seed', type=int, default=100,
                    help='random seed')
parser.add_argument('--basemodel', type=int, default=100,
                    help='base model')

args = parser.parse_args()

set_seed(args.seed)
os.environ["WANDB_MODE"] = "dryrun"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

log_path = './logs/{}'.format('All/Basemodel_{}/pretrain_{}/seed_{}'.format(args.basemodel, args.pretrain, args.seed))
summary = tensorboardX.SummaryWriter(log_path)

if args.basemodel == 1:
    net = Resnet(args.pretrain > 0, 128)
elif args.basemodel == 2:
    net = VggNet(args.pretrain > 0, 128)
elif args.basemodel == 3:
    net = GoogleNet(args.pretrain > 0, 128)
elif args.basemodel == 4:
    net = InceptionNet(args.pretrain > 0, 128)
else:
    net = BaseModel(3, 128, [224, 224], num_layers=5)

multi_header = MultiHeadClassifier(128, 6, 7)
measure = MeasureModel(128)


train_data = NoiseValOrganoid('all_samples.h5')
test_data = NoiseValOrganoid('validation_samples.h5')
train_loader = DataLoader(train_data, args.batchsize, True, num_workers=1)
test_loader = DataLoader(test_data, 8, False, num_workers=1)
trainer = Adam(chain(net.parameters(), multi_header.parameters(), measure.parameters()), 0.001, betas=(0.5, 0.99),
               weight_decay=0.001)
lr_schedule = StepLR(trainer, 10, 0.95)
loss_f = CrossEntropyLoss()


def calc_dist_matrix(queue_feature, gallery_feature):
    queue_feature_norm = (queue_feature ** 2).sum(1).view(-1, 1)
    gallery_feature_norm = (gallery_feature ** 2).sum(1).view(1, -1)
    dist = queue_feature_norm + gallery_feature_norm - \
           2.0 * torch.mm(queue_feature, torch.transpose(gallery_feature, 0, 1))
    return dist


def calc_sign(label):
    a = torch.zeros(label.size(0), label.size(0))
    for i in range(label.size(0)):
        for j in range(label.size(0)):
            value_pos = torch.sum((label[i].float() - label[j].float()) > 0).float()
            value_neg = torch.sum((label[i].float() - label[j].float()) < 0).float()
            value_equ = torch.sum((label[i].float() - label[j].float()) == 0).float()
            if value_pos >= 4:
                a[i, j] = 1
            if value_neg >= 4:
                a[i, j] = -1
            if value_equ >= 4:
                a[i, j] = 0
    return a


def calc_sign_V2(label):
    a = torch.zeros(label.size(0), label.size(0))
    for i in range(label.size(0)):
        for j in range(label.size(0)):
            value_pos = torch.sum((label[i].float() - label[j].float()) > 0).float()
            value_neg = torch.sum((label[i].float() - label[j].float()) < 0).float()
            value_equ = torch.sum((label[i].float() - label[j].float()) == 0).float()
            if value_pos >= 1:
                a[i, j] = 1
            if value_neg >= 1:
                a[i, j] = -1
            if value_equ >= 1:
                a[i, j] = 0
    return a


def calc_sign_V3(label, worker):
    a = torch.zeros(label.size(0), label.size(0))
    for i in range(label.size(0)):
        for j in range(i, label.size(0)):
            value = label[i] - label[j]
            if worker[i] == worker[j]:
                if value > 0:
                    a[i, j] = 1
                if value < 0:
                    a[i, j] = -1
                if value == 0:
                    a[i, j] = 0
            else:
                if value > 1:
                    a[i, j] = 1
                if value < -1:
                    a[i, j] = -1
    return a


if __name__ == '__main__':
    best_train_acc = 0
    best_val_acc = 0
    for epoch in range(args.epoch):
        net.train()
        multi_header.train()
        measure.train()
        train_loss = []
        Y = [[] for i in range(7)]
        Y_ = [[] for i in range(7)]
        Y_prob = [[] for i in range(7)]
        for i, batch in enumerate(train_loader):
            x = batch['image'].float()
            y = batch['category'].long()
            w = batch['width'].float().view(-1, 1)
            h = batch['height'].float().view(-1, 1)
            worker_id = batch['worker'].long()
            mask = torch.eye(y.size(0)).float()
            sign = calc_sign_V3(y, worker_id)

            if torch.cuda.is_available():
                net = net.cuda()
                multi_header = multi_header.cuda()
                measure = measure.cuda()
                x = x.cuda()
                y = y.cuda()
                w = w.cuda()
                h = h.cuda()
                worker_id = worker_id.cuda()
                mask = mask.cuda()
                sign = sign.cuda()

            features = net(x, w, h)
            y_prob = multi_header(features)


            margin = 0
            measure_val = measure(features)
            measure_arr = measure_val - measure_val.t()
            noise = torch.randn_like(measure_arr).to(measure_arr.device) * 0.05 / (epoch // 25 + 1)
            measure_arr = measure_arr + noise
            print(measure_val.squeeze())
            measure_mask = (((measure_arr * sign) < 0).float().detach() + (
                    (measure_arr == 0) * (sign != 0)).float().detach()) % 2  

            masked_measure = torch.sum(
                (-measure_mask * (sign > 0).float() + measure_mask * (
                        sign < 0).float()) * measure_arr) / torch.sum(
                (measure_mask != 0).float() + 1e-5) 

            regular_term = torch.mean(measure_val - measure_val ** 2) * ((torch.sum(torch.abs(
                measure_mask.detach())) > 10).float())

            dist = calc_dist_matrix(features, features)
            p = (dist / 5) / (torch.sum(dist / 5 * (1 - mask)))
            p = p * (1 - mask) + mask
            p = p * torch.log(p + 1e-5)
            delta = 0  
            loss = -torch.mean(torch.sum(p, -1)) + masked_measure + regular_term * 0.005
            
            print(float(torch.sum(torch.abs(sign))), float(torch.sum(torch.abs(measure_mask))),
                  float(masked_measure), float(regular_term))
            for worker in range(7):
                if not torch.any(worker_id == worker):
                    continue
                y_ = y[worker_id == worker]
                y_prob_ = y_prob[:, worker, :]
                y_prob_ = y_prob_[worker_id == worker]
                loss = loss + loss_f(y_prob_, y_)

            train_loss.append(float(loss))
            trainer.zero_grad()
            loss.backward()
            trainer.step()

            print('Train, epoch:{}, iter:{}, loss:{}'.format(epoch, i, float(loss)))

            for worker in range(7):
                if not torch.any(worker_id == worker):
                    continue
                y_prob_ = y_prob[:, worker, :]
                y_prob_ = y_prob_[worker_id == worker]
                y_prob_ = torch.softmax(y_prob_, -1)
                Y_prob[worker].append(y_prob_.detach().cpu().numpy())

                y_ = torch.argmax(y_prob_, -1)
                Y_[worker].append(y_.detach().cpu().numpy())
                Y[worker].append(y[worker_id == worker].cpu().numpy())

        for worker in range(7):
            Y[worker] = np.concatenate(Y[worker], 0)
            Y_prob[worker] = np.concatenate(Y_prob[worker], 0)
            Y_[worker] = np.concatenate(Y_[worker], 0)

        Y = np.concatenate(Y, 0)
        Y_prob = np.concatenate(Y_prob, 0)
        Y_ = np.concatenate(Y_, 0)
        train_acc = np.mean((Y == Y_) * 1.0)
        best_train_acc = train_acc if train_acc > best_train_acc else best_train_acc

        summary.add_scalar('lr', trainer.param_groups[0]['lr'], epoch)

        np.save(log_path + '/train_label_{}.npy'.format(epoch), Y)
        np.save(log_path + '/train_pre_{}.npy'.format(epoch), Y)
        torch.cuda.empty_cache()

        summary.add_scalars('train_acc',
                            {'worker{}'.format(worker): np.mean((Y[worker] == Y_[worker]) * 1.0) for worker in
                             range(7)}, epoch)

        net.eval()
        multi_header.eval()
        measure.eval()

        val_loss = [[[] for i in range(7)]]
        Y = [[] for i in range(7)]
        Y_ = [[] for i in range(7)]
        Y_prob = [[] for i in range(7)]
        images = [[] for i in range(7)]
        ALL_image = []
        Features = []
        Label = []
        Measure = []
        W = []
        H = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x = batch['image'].float()
                y = batch['category'].long()
                w = batch['width'].float().view(-1, 1)
                h = batch['height'].float().view(-1, 1)

                if torch.cuda.is_available():
                    net = net.cuda()
                    multi_header = multi_header.cuda()
                    measure = measure.cuda()
                    x = x.cuda()
                    y = y.cuda()
                    w = w.cuda()
                    h = h.cuda()

                features = net(x, w, h)
                measure_val = measure(features)
                y_prob = multi_header(features)
                loss = 0

                for worker in range(7):
                    y_prob_ = y_prob[:, worker, :]
                    loss = loss + loss_f(y_prob_, y)

                print('Val, epoch:{}, iter:{}, loss:{}'.format(epoch, i, float(loss)))

                for worker in range(7):
                    y_prob_ = y_prob[:, worker, :]
                    y_prob_ = torch.softmax(y_prob_, -1)
                    Y_prob[worker].append(y_prob_.detach().cpu().numpy())
                    y_ = torch.argmax(y_prob_, -1)
                    Y_[worker].append(y_.detach().cpu().numpy())
                    Y[worker].append(y.cpu().numpy())
                    images[worker].append(x[y_ != y])
                Features.append(features.cpu())
                ALL_image.append(x.cpu())
                Label.append(y.cpu())
                Measure.append(measure_val.cpu())
                W.append(w.cpu())
                H.append(h.cpu())

        for worker in range(7):
            Y[worker] = np.concatenate(Y[worker], 0)
            Y_prob[worker] = np.concatenate(Y_prob[worker], 0)
            Y_[worker] = np.concatenate(Y_[worker], 0)
            images[worker] = torch.cat(images[worker], 0)

        Features = torch.cat(Features, 0)
        ALL_image = torch.cat(ALL_image, 0)
        Label = torch.cat(Label, 0)
        Measure = torch.cat(Measure, 0)
        W = torch.cat(W, 0)
        H = torch.cat(H, 0)
        np.save(log_path + '/test_label_{}.npy'.format(epoch), Label)
        np.save(log_path + '/test_pre_{}.npy'.format(epoch), Y)
        np.save(log_path + '/measure_{}.npy'.format(epoch), Measure)
        np.save(log_path + '/feature_{}.npy'.format(epoch), Features)
        np.save(log_path + '/w_{}.npy'.format(epoch), W)
        np.save(log_path + '/h_{}.npy'.format(epoch), H)

        
        fig = plt.figure()

        ax = fig.add_subplot(221)
        plt.plot(np.arange(Measure.size(0)), Measure.numpy())

        ax = fig.add_subplot(222)
        plt.plot(np.arange(Label.size(0)), Label.numpy())

        ax = fig.add_subplot(223)
        plt.plot(np.arange(W.size(0)), W.numpy())

        ax = fig.add_subplot(224)
        plt.plot(np.arange(H.size(0)), H.numpy())

        summary.add_figure('measure', fig, epoch)
        summary.add_scalars('val_acc',
                            {'worker{}'.format(worker): np.mean((Y[worker] == Y_[worker]) * 1.0) for worker in
                             range(7)}, epoch)

        m = calc_sign_V2(Measure).cpu().numpy()
        s = calc_sign_V2(Label).cpu().numpy()
        w = calc_sign_V2(W).cpu().numpy()
        h = calc_sign_V2(H).cpu().numpy()
        wh = calc_sign_V2(W * H).cpu().numpy()
        label_mask = (s != 0)

        summary.add_scalars('val', {
            'p_v': float(np.mean((m[label_mask] == s[label_mask]) * 1.0)),
            'w_v': float(np.mean((w[label_mask] == s[label_mask]) * 1.0)),
            'h_v': float(np.mean((h[label_mask] == s[label_mask]) * 1.0)),
            'wh_v': float(np.mean((wh[label_mask] == s[label_mask]) * 1.0)),
            'p_w': float(np.mean((m[label_mask] == w[label_mask]) * 1.0)),
            'p_h': float(np.mean((m[label_mask] == h[label_mask]) * 1.0)),
            'p_wh': float(np.mean((m[label_mask] == wh[label_mask]) * 1.0)),
        }, epoch)

        lr_schedule.step()
        summary.add_scalars('best_acc', {'train': best_train_acc}, epoch)
        summary.add_embedding(Features, global_step=epoch)
    summary.close()
