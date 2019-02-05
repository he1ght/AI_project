import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


class MNISTFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, shallow=False):
        super(MNISTFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module('conv1', nn.Conv2d(1, 64, kernel_size=15, padding=1, stride=5))
        else:
            self.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
            self.add_module('relu1', nn.ReLU())
            self.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop1', nn.Dropout(dropout_rate))
            self.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.add_module('relu2', nn.ReLU())
            self.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop2', nn.Dropout(dropout_rate))
            self.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.add_module('relu3', nn.ReLU())
            self.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            self.add_module('drop3', nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        if self.shallow:
            return 64 * 4 * 4
        else:
            return 128 * 3 * 3


class UCIStudPerformLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., hidden_features=1024, out_features=1024, shallow=True):
        # 1st model
        super(UCIStudPerformLayer, self).__init__()
        self.shallow = shallow
        self.feature_hidden = hidden_features  # 1024
        self.feature_out = out_features  # 1024
        if shallow:
            self.add_module('linear', nn.Linear(80, self.feature_out))
            #self.add_module('batch_norm', nn.BatchNorm1d(self.feature_out))
            self.add_module('relu', nn.ReLU())
            self.add_module('dropout', nn.Dropout(dropout_rate))
        else:
            # raise NotImplementedError
            self.add_module('linear1', nn.Linear(80, self.feature_hidden))
            self.add_module('relu1', nn.ReLU())
            self.add_module('dropout1', nn.Dropout(dropout_rate))
            self.add_module('linear2', nn.Linear(self.feature_hidden, self.feature_hidden))
            self.add_module('relu2', nn.ReLU())
            self.add_module('dropout2', nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        return self.feature_out


class UCIStudLayer(nn.Sequential):
    def __init__(self, dropout_rate=0., hidden_features=1024, out_features=1024, shallow=True):
        # 2nd model
        super(UCIStudLayer, self).__init__()
        self.shallow = shallow
        self.feature_hidden = hidden_features  # 1024
        self.feature_out = out_features  # 1024
        if shallow:
            self.add_module('linear', nn.Linear(238, self.feature_out))
            self.add_module('bn', nn.BatchNorm1d(self.feature_out))
            self.add_module('relu', nn.ReLU())
            self.add_module('dropout', nn.Dropout(dropout_rate))
        else:
            # raise NotImplementedError
            self.add_module('linear1', nn.Linear(238, self.feature_hidden))
            self.add_module('bn1', nn.BatchNorm1d(self.feature_hidden))
            self.add_module('relu1', nn.ReLU())
            self.add_module('dropout1', nn.Dropout(dropout_rate))
            self.add_module('linear2', nn.Linear(self.feature_hidden, self.feature_out))
            self.add_module('bn2', nn.BatchNorm1d(self.feature_out))
            self.add_module('relu2', nn.ReLU())
            self.add_module('dropout2', nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        return self.feature_out


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, dropout_rate=0.3):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # leaf label distribution
        self.pi = np.ones((self.n_leaf, n_class)) / n_class
        self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
            ('dropout', nn.Dropout(dropout_rate)),
        ]))

    def forward(self, x):
        feats = torch.mm(x, self.feature_mask)                  # [batch_size,n_used_feature]
        decision = self.decision(feats)                         # [batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # [batch_size,n_leaf,2]

        # compute route probability
        #  2^n - 1 count : [1:2^n]
        batch_size = x.size(0)
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.), requires_grad=True)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]          # [batch_size,2**n_layer,2]
            _mu = _mu * _decision                                  # [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)
        return mu

    def get_pi(self):
        return F.softmax(self.pi, dim=-1)

    def calculate_probability(self, mu, pi):
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, dropout_rate):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        self.n_class = n_class
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, dropout_rate=dropout_rate)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.calculate_probability(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob


class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super(NeuralDecisionForest, self).__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size(0), -1)
        out = self.forest(out)
        return out
