#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Loss.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/11/24 17:27
# @Desc  :

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):

    def __init__(self, category=6):
        """Constructor for MAELoss"""
        super(MAELoss, self).__init__()
        self.loss_f = nn.L1Loss()
        self.category = category

    def forward(self, pre, target):
        pre_log = F.softmax(pre, -1)
        target_one_hot = F.one_hot(target, self.category)
        return self.loss_f(pre_log, target_one_hot)


class GCELoss(nn.Module):
    """"""

    def __init__(self, category=6):
        """Constructor for GCELoss"""
        super(GCELoss, self).__init__()
        self.category = category

    def forward(self, pre, target, q=0.5):
        pre_log = F.softmax(pre, -1)
        target_one_hot = F.one_hot(target, self.category)
        ans = (target_one_hot - torch.pow(pre_log + 1e-5, q)) * target_one_hot / q
        ans = torch.mean(ans)
        return ans


if __name__ == '__main__':
    import torch
    import numpy as np

    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    pre = torch.randn(5, 10)
    target = torch.tensor([0, 1, 9, 5, 8])
    for i in np.linspace(0.1, 1, 10):
        pre = torch.randn(5, 10)

        y_1_ = torch.argmax(pre, -1)
        y_2_ = torch.argmax(pre, -1)
        print(y_1_)
        print(y_2_)
        if torch.any(y_1_ != y_2_):
            print(i, torch.mean(loss_f(pre, target)[y_1_ != y_2_]))
        else:
            print(0)
