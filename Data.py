#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Data.py
# @Author: Xuesheng Bian
# @Email: xbc0809@gmail.com
# @Date  :  2020/8/3 15:37
# @Desc  :

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ToPILImage, ToTensor, Normalize, \
    Resize
import random


class Organoid(Dataset):
    """"""

    def __init__(self, path):
        """Constructor for Organoid"""
        super(Organoid, self).__init__()
        self.h5_data = h5py.File(path, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        image = self.h5_data['images'][item]
        label = self.h5_data['box_info'][item]
        return {
            'image': self.transform(image),
            'position': label[1:],
            'category': label[0],
        }

    def __len__(self):
        return self.h5_data['images'].shape[0]


class Organoid_classification(Dataset):
    """"""

    def __init__(self, path, train=True):
        """Constructor for Organoid_classification"""
        super(Organoid_classification, self).__init__()
        file = h5py.File(path, 'r')
        self.data = file['train'] if train else file['test']
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.train = train
        if self.train:
            self.selected = self._get_index(self.data)
        else:
            self.selected = list(range(len(self.data['category'])))

    def _get_index(self, data):
        index_array = [[] for i in range(len(set(self.data['category'])))]
        for i in range(len(data['category'])):
            category = data['category'][i]
            index_array[category].append(i)
        l = np.max([len(x) for x in index_array])
        for x in index_array:
            l_x = len(x)
            while len(x) != l:
                x.append(x[np.random.randint(0, l_x)])
            random.shuffle(x)
        return np.stack(index_array).flatten(order='F')

    def __getitem__(self, item):
        select_index = self.selected[item]
        category = self.data['category'][select_index]
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = np.stack([self.data['image'][select_index]] * 3, -1)
        return {
            'category': category,
            'image': self.transform(img),
            'width': width,
            'height': height,
        }

    def __len__(self):
        if self.train:
            return self.selected.shape[0]
        else:
            return len(self.data['category'])


class NoisePrivateOrganoid(Dataset):
    """"""

    def __init__(self, path):
        """Constructor for Noise_labelled_Organoid"""
        super(NoisePrivateOrganoid, self).__init__()
        self.data = h5py.File(path, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.selected = self._get_index(self.data)

    def _get_index(self, data):
        index_array = [[] for i in range(len(set(self.data['category'])))]
        for i in range(len(data['category'])):
            category = data['category'][i]
            index_array[category].append(i)
        l = np.max([len(x) for x in index_array])
        for x in index_array:
            l_x = len(x)
            while len(x) != l:
                x.append(x[np.random.randint(0, l_x)])
            random.shuffle(x)
        return np.stack(index_array).flatten(order='F')

    def __getitem__(self, item):
        select_index = self.selected[item]
        category = self.data['category'][select_index].astype(np.uint8)
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = self.data['image'][select_index]
        return {
            'category': category,
            'image': self.transform(img),
            'width': width,
            'height': height,
        }

    def __len__(self):
        return self.selected.shape[0]


class NoiseSharedOrganoid(Dataset):
    """"""

    def __init__(self, path, worker=1):
        """Constructor for Noise_labelled_Organoid"""
        super(NoiseSharedOrganoid, self).__init__()
        self.data = h5py.File(path, 'r')
        self.worker = worker
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.selected = self._get_index(self.data)

    def _get_index(self, data):
        index_array = [[] for i in range(len(set(self.data['category'][:, self.worker])))]
        for i in range(len(data['category'])):
            category = data['category'][:, self.worker][i] if not self.worker < 0 else \
                np.median(data['category'], 1).flatten()[i]
            index_array[int(category)].append(i)
        l = np.max([len(x) for x in index_array])
        for x in index_array:
            l_x = len(x)
            while len(x) != l:
                x.append(x[np.random.randint(0, l_x)])
            random.shuffle(x)
        return np.stack(index_array).flatten(order='F')

    def __getitem__(self, item):
        select_index = self.selected[item]
        category = self.data['category'][select_index].astype(np.uint8)
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = self.data['image'][select_index]
        return {
            'category': category[self.worker] if not self.worker < 0 else np.median(category),
            'image': self.transform(img),
            'width': width,
            'height': height,
            'worker': self.worker,
        }

    def __len__(self):
        return self.selected.shape[0]


class NoiseValOrganoid(Dataset):
    """"""

    def __init__(self, path):
        """Constructor for Noise_labelled_Organoid"""
        super(NoiseValOrganoid, self).__init__()
        self.data = h5py.File(path, 'r')
        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, item):
        select_index = item
        category = self.data['category'][select_index].astype(np.uint8)
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = self.data['image'][select_index]

        try:
            worker = self.data['worker'][select_index]
        except Exception as e:
            worker = 0

        return {
            'category': category,
            'image': self.transform(img),
            'width': width,
            'height': height,
            'worker': worker,
        }

    def __len__(self):
        return len(self.data['category'])


class NoiseConsensusOrganoid(Dataset):
    """"""

    def __init__(self, path):
        """Constructor for Noise_labelled_Organoid"""
        super(NoiseConsensusOrganoid, self).__init__()
        self.data = h5py.File(path, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.consensus_id = self._get_consensus()

    def _get_consensus(self):
        category = self.data['category'][...]
        std = np.std(category, 1)
        indices = np.arange(len(category))[std == 0]
        selected_category = category[indices.tolist(), 0]
        index_array = [[] for i in range(len(set(selected_category)))]
        for i in range(len(selected_category)):
            index_array[int(selected_category[i])].append(indices[i])
        l = np.max([len(x) for x in index_array])
        for x in index_array:
            l_x = len(x)
            while len(x) != l:
                x.append(x[np.random.randint(0, l_x)])
            random.shuffle(x)
        return np.stack(index_array).flatten(order='F')

    def __getitem__(self, item):
        select_index = self.consensus_id[item]
        category = self.data['category'][select_index, 0].astype(np.uint8)
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = self.data['image'][select_index]
        return {
            'category': category,
            'image': self.transform(img),
            'width': width,
            'height': height,
        }

    def __len__(self):
        return len(self.consensus_id)


class NoiseOrganoid(Dataset):
    """"""

    def __init__(self, path):
        """Constructor for NoiseOrganoid"""
        super(NoiseOrganoid, self).__init__()
        self.data = h5py.File(path, 'r')
        self.transform = Compose([
            ToPILImage(),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.selected = self._get_index(self.data)

    def _get_index(self, data):
        index_array = [[] for i in range(len(set(self.data['category'])))]
        for i in range(len(data['category'])):
            category = data['category'][i]
            index_array[category].append(i)
        l = np.max([len(x) for x in index_array])
        for x in index_array:
            l_x = len(x)
            while len(x) != l:
                x.append(x[np.random.randint(0, l_x)])
            random.shuffle(x)
        return np.stack(index_array).flatten(order='F')

    def __getitem__(self, item):
        select_index = self.selected[item]
        category = self.data['category'][select_index].astype(np.uint8)
        width = self.data['w_h'][select_index][0] * 1.0 / self.data['wid_hei'][select_index][0]
        height = self.data['w_h'][select_index][1] * 1.0 / self.data['wid_hei'][select_index][1]
        img = self.data['image'][select_index]

        try:
            worker = self.data['worker'][select_index]
        except Exception as e:
            worker = 0

        return {
            'category': category,
            'image': self.transform(img),
            'width': width,
            'height': height,
            'worker': worker,
        }

    def __len__(self):
        return len(self.data['category'])


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.models import resnet50
    from torch.nn import CrossEntropyLoss
    import torch
    import cv2
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    set_seed(100)

    # data = Organoid_classification('../classification_split.h5', True)
    data = NoiseValOrganoid('validation_samples.h5')
    # data = NoiseValOrganoid('shared_samples.h5')
    # data = NoisePrivateOrganoid('private_samples_1.h5')
    # data = NoiseSharedOrganoid('shared_samples.h5', 1)
    # data = NoiseConsensusOrganoid('shared_samples.h5')
    data = NoiseOrganoid('noise_organoid.h5')
    data = NoiseOrganoid('all_samples.h5')
    loader = DataLoader(data, 20, True)
    net = resnet50(False, num_classes=6)
    loss_f = CrossEntropyLoss()
    Y = []
    Y_ = []
    Y_prob = []
    for i, batch in enumerate(loader):
        image = batch['image'].float()
        # w = batch['width']
        # h = batch['width']
        # print(batch['width'])
        # print(batch['width'])
        print(batch['worker'])

        # m = calc_sign_V2(batch['width'] * batch['height']).cpu().numpy()
        # # m = calc_sign_V2(batch['height']).cpu().numpy()
        # s = calc_sign_V2(batch['category']).cpu().numpy()
        # label_mask = (s != 0)
        #
        # val = np.mean((m[label_mask] == s[label_mask]) * 1.0)
        # print('val', val)

        # w = batch['width'].float().view(-1, 1)
        # h = batch['height'].float().view(-1, 1)
        # print(w)
        # print(h)
        # sign = torch.zeros(batch['category'].size(0), batch['category'].size(0))
        # for i in range(sign.size(0)):
        #     for j in range(sign.size(1)):
        #         value_pos = torch.sum((batch['category'][i].float() - batch['category'][j].float()) > 0).float()
        #         value_neg = torch.sum((batch['category'][i].float() - batch['category'][j].float()) < 0).float()
        #         value_equ = torch.sum((batch['category'][i].float() - batch['category'][j].float()) == 0).float()
        #         if value_pos >= 1:
        #             sign[i, j] = 1
        #         if value_neg >= 1:
        #             sign[i, j] = -1
        #         if value_equ >= 1:
        #             sign[i, j] = 0
        # print(sign)
        # print(sign.size())
        # sign = sign.numpy()
        # print(sign[sign != 0])
        # print(np.sum(np.abs(sign[sign != 0])))
        # measure_val = torch.randn(10, 1)
        # measure_arr = measure_val - measure_val.t()
        # measure_mask = ((measure_arr * sign) < 0).float()
        # masked_measure = torch.sum(
        #     (measure_mask * (sign > 0).float() - measure_mask * (sign < 0).float()) * measure_arr) / torch.sum(
        #     (measure_mask != 0).float())

        # print(measure_arr)
        # print(sign)
        # print(measure_mask)
        # print(torch.sum((measure_mask != 0).float()), masked_measure)

        # for i in range(batch['category'].size(1)):
        #     x_arr = np.arange(batch['category'].size(0))
        #     y_arr = batch['category'][:, i].numpy()
        #     plt.plot(x_arr, y_arr)
        # plt.show()

        # image = make_grid(image[0, 0], normalize=True).squeeze().numpy().transpose([1, 2, 0])
        # image = cv2.resize(image, (h, w))
        # cv2.imshow(str(int(batch['category'])), image)
        # cv2.waitKey()
