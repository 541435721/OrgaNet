#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Models.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/8/3 15:37
# @Desc  :

import torch.nn as nn
import numpy as np
import random
import torch
from torchvision.models import googlenet, resnet50, inception_v3, vgg13


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GoogleNet(nn.Module):
    """"""

    def __init__(self, pretrain=False, num_classes=11):
        """Constructor for GoogleNet"""
        super(GoogleNet, self).__init__()
        self.basenet = googlenet(pretrain, aux_logits=False)
        self.basenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x, w=None, h=None):
        out = self.basenet(x)

        if w is not None:
            out[:, -1] = w[:, 0]
        if h is not None:
            out[:, -2] = h[:, 0]

        return out


class Resnet(nn.Module):
    """"""

    def __init__(self, pretrain=False, num_classes=11):
        """Constructor for Resnet"""
        super(Resnet, self).__init__()
        self.basenet = resnet50(pretrain)
        self.basenet.fc = nn.Linear(2048, num_classes)

    def forward(self, x, w=None, h=None):
        out = self.basenet(x)

        if w is not None:
            out[:, -1] = w[:, 0]
        if h is not None:
            out[:, -2] = h[:, 0]

        return out


class InceptionNet(nn.Module):
    """"""

    def __init__(self, pretrain=False, num_classes=11):
        """Constructor for InceptionNet"""
        super(InceptionNet, self).__init__()
        self.basenet = inception_v3(pretrain, aux_logits=False)
        self.basenet.fc = nn.Linear(2048, num_classes)

    def forward(self, x, w=None, h=None):
        out = self.basenet(x)

        if w is not None:
            out[:, -1] = w[:, 0]
        if h is not None:
            out[:, -2] = h[:, 0]

        return out


class VggNet(nn.Module):
    """"""

    def __init__(self, pretrain=False, num_classes=11):
        """Constructor for VggNet"""
        super(VggNet, self).__init__()
        self.basenet = vgg13(pretrain)
        self.basenet.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x, w=None, h=None):
        out = self.basenet(x)

        if w is not None:
            out[:, -1] = w[:, 0]
        if h is not None:
            out[:, -2] = h[:, 0]

        return out


class HybridNet(nn.Module):
    """"""

    def __init__(self, pretrain=False, num_classes=11, basenet='vgg'):
        """Constructor for HybridNet"""
        super(HybridNet, self).__init__()
        if basenet == 'vgg':
            self.basenet = vgg13(pretrain, )
            self.basenet.classifier[-1] = nn.Identity()
            self.last = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(4098, num_classes))
        elif basenet == 'resnet':
            self.basenet = resnet50(pretrain)
            self.basenet.fc = nn.Identity()
            self.last = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2050, num_classes))
        elif basenet == 'googlenet':
            self.basenet = googlenet(pretrain, aux_logits=False)
            self.basenet.fc = nn.Identity()
            self.last = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(1026, num_classes))
        elif basenet == 'inception':
            self.basenet = inception_v3(pretrain, aux_logits=False)
            self.basenet.fc = nn.Identity()
            self.last = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(2050, num_classes))

    def forward(self, x, w, h):
        feature = self.basenet(x)
        feature = torch.cat([feature, w, h], -1)
        return self.last(feature)


class BaseModel(nn.Module):
    """"""

    def __init__(self, in_channels, out_channels, input_size, increase_factor=16, num_layers=6):
        """Constructor for BaseModel"""
        super(BaseModel, self).__init__()
        self.backbone = self._make_layer(in_channels, increase_factor, num_layers)
        self.refine = nn.Conv2d(increase_factor * (2 ** (num_layers - 1)), out_channels,
                                input_size[-1] // (2 ** num_layers))

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, in_channels, increase_factor, num_layers):
        layers = []
        input_channels = in_channels
        for i in range(num_layers):
            layer = self._make_block(input_channels, increase_factor * (2 ** i))
            input_channels = increase_factor * (2 ** i)
            layers.append(layer)
            pool = nn.MaxPool2d(2, 2)
            layers.append(pool)
        return nn.Sequential(*layers)

    def forward(self, x, w=None, h=None):
        out = self.backbone(x)
        out = self.refine(out)
        out = out.squeeze(dim=-1).squeeze(dim=-1)
        # 量纲归一化
        if w is not None:
            out[:, -1] = w[:, 0]
        if h is not None:
            out[:, -2] = h[:, 0]

        return out


class MultiHeadClassifier(nn.Module):
    """"""

    def __init__(self, in_channels, category, headers=10):
        """Constructor for MultiHeadClassifier"""
        super(MultiHeadClassifier, self).__init__()
        self.multi_header = nn.ModuleList()
        for i in range(headers):
            self.multi_header.append(self._make_layer(in_channels, category))

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, out_channels),
        )

    def forward(self, x):
        out = []
        for layer in self.multi_header:
            out.append(layer(x))
        out = torch.stack(out, 1)
        return out


class Classifier(nn.Sequential):
    """"""

    def __init__(self, in_channels, out_channels):
        """Constructor for Classifier"""
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, out_channels),
        )

    def forward(self, x):
        return super(Classifier, self).forward(x)


class MeasureModel(nn.Module):
    """"""

    def __init__(self, in_channels):
        """Constructor for MeasureModel"""
        super(MeasureModel, self).__init__()
        self.measure = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.measure(x)


class Transition_Layer(nn.Sequential):
    """"""

    def __init__(self, in_channels, workers, category):
        """Constructor for Transition_Layer"""
        super(Transition_Layer, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels // 2, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, workers * category * category),
        )

    def forward(self, x):
        return super(Transition_Layer, self).forward(x)


if __name__ == '__main__':
    set_seed(0)
    x = torch.randn(2, 3, 224, 224)
    w = torch.randn(2, 1)
    h = torch.randn(2, 1)
    net = BaseModel(3, 128, x.size(), num_layers=5)
    net = Resnet(False, 128)
    net = VggNet(False, 128)
    net = GoogleNet(False, 128)
    net = InceptionNet(False, 128)
    y = net(x)
    print(y.size())
    multi_header = Transition_Layer(128, 7, 6)
    # y = multi_header(y)
    # print(y.size())
