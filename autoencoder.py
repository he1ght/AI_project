import argparse
import logging
import pickle

import torch
import torch.nn as nn
import dataset
import torch.nn.functional as F
import numpy as np


def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='autoencoder.py')
    parser.add_argument('-dataset', choices=['1st', '2nd'], default='1st')
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-save', type=str, default="model/test")

    opt = parser.parse_args()
    return opt


class net(nn.Module):
    def __init__(self, input_feature, hidden_feature):
        super().__init__()
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature

        self.fc_in = nn.Linear(input_feature, hidden_feature)
        self.fc_out = nn.Linear(hidden_feature, input_feature)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_out(x)
        return x


def train(opt, cases):
    return

    train_data = dataset.CV_SAP_data_set("./data/1st", cv_index=cv_idx)
    test_data = dataset.CV_SAP_data_set("./data/1st", cv_index=cv_idx, train=False)
    data_len = len(train_data)

    model = net(80, 10, 3, layers=0).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    cri = torch.nn.MSELoss()

    epochs = 100
    report_every = 2
    batch_size = 64
    last = ()

    for epoch in range(epochs):
        model.train()
        bloss = 0.0
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for i, (x, y) in enumerate(train_loader):
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.nll_loss(torch.log(pred), y, size_average=False)
            bloss += loss.item()
            loss.backward()
            optim.step()
        # print("Epoch {:3d} : loss {}".format(epoch + 1, bloss/data_len))

        model.eval()
        bloss = 0.0
        correct = 0
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            bloss += F.nll_loss(torch.log(output), y, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
        test_len = len(test_data)
        print("[Test] Epoch {:3d} : loss {} [ {} / {} ( {:.2f} % )]".format(epoch + 1, bloss / data_len, correct,
                                                                            test_len,
                                                                            float(float(correct) / float(
                                                                                test_len)) * 100))
        last = (correct, test_len)
    print()
    return last


def load_obj(name, path="."):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def preprocess(opt):
    if opt.dataset == '1st':
        root = "./data/1st"
    elif opt.dataset == '2nd':
        root = "./data/2nd"
    else:
        raise NotImplemented
    fields_x = load_obj("fields", root)
    X = None

    # for field in fields_x:
    #     a = np.eye(field['len'])
    #     if X is not None:
    #         b = torch.from_numpy(np.repeat(a, X.size(0), axis=0))
    #         X = X.repeat(b.size(1), 1)
    #         X = torch.cat((X, b), dim=1)
    #     else:
    #         X = torch.from_numpy(a)
    #     print(X.size())
    # batchs = []
    # idx = 0
    # while True:
    #     e_idx = idx + opt.batch_size
    #     batchs += [X[idx:e_idx]]
    #     if e_idx >= len(X):
    #         break
    # print(len(batchs))

    case_minmax = []
    cnt = 1
    for field in fields_x:
        case_minmax.append([0, field['len']])
        cnt *= field['len']

    def travel(case_minmax, cases, next_case):
        if not case_minmax:
            cases.append(next_case)
            print(len(cases))
            return
        for case in range(case_minmax[0][0], case_minmax[0][1]):
            new_case = next_case + [case]
            travel(case_minmax[1:], cases, new_case)

    cases = []
    travel(case_minmax, cases, [])
    print(cases)
    return case_minmax


def main():
    opt = parse_arg()

    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
        opt.device = 'cuda:{}'.format(opt.gpuid)
    else:
        opt.device = 'cpu'

    cases = preprocess(opt)
    train(opt, cases)


if __name__ == "__main__":
    main()
