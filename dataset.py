import os
import pickle

import torch
import torchvision
from torch.utils.data import Dataset


def load_obj(name, path="."):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class SAP_data_set(Dataset):
    def __init__(self, root, k_fold=10, train=True):
        super(SAP_data_set, self).__init__()
        self.root = os.path.expanduser(root)
        self.k_fold = k_fold
        self.train = train

        self.data_range = list(range(self.k_fold))

        self.X = []
        self.Y = []

        self.fields_x = load_obj("fields", self.root)
        self.fields_y = load_obj("atd", self.root)

        for ri in self.data_range:
            cur_cv_x_data = torch.load(os.path.join(self.root, "data.{}.pt".format(ri)))
            self.X += [cur_cv_x_data]
            cur_cv_y_data = torch.load(os.path.join(self.root, "result.{}.pt".format(ri)))
            self.Y += [cur_cv_y_data]
        self.X = torch.cat(self.X)
        self.Y = torch.cat(self.Y)
        # self.X = torch.cat(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return len(self.X)


class CV_SAP_data_set(Dataset):
    def __init__(self, root, k_fold=10, cv_index=0, train=True):
        super(CV_SAP_data_set, self).__init__()
        self.root = os.path.expanduser(root)
        self.k_fold = k_fold
        self.train = train
        self.set_idx = cv_index

        if self.train:
            self.data_range = list(range(self.k_fold))
            self.data_range.remove(self.set_idx)
        else:
            self.data_range = [self.set_idx]

        self.X = []
        self.y = []

        self.fields_x = load_obj("fields", self.root)
        self.fields_y = load_obj("atd", self.root)

        for ri in self.data_range:
            cur_cv_x_data = torch.load(os.path.join(self.root, "data.{}.pt".format(ri)))
            self.X += [cur_cv_x_data]
            cur_cv_y_data = torch.load(os.path.join(self.root, "result.{}.pt".format(ri)))
            self.y += [cur_cv_y_data]
        self.X = torch.cat(self.X)
        self.y = torch.cat(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
        # return (self.X[index].div(torch.norm(self.X[index], 2)), self.y[index])

    def __len__(self):
        return len(self.X)