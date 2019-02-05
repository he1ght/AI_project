import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset


class net(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature, layers=0):
        super().__init__()
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.output_feature = output_feature

        self.fc_in = nn.Linear(input_feature, hidden_feature)
        self.fc_out = nn.Linear(hidden_feature, output_feature)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(hidden_feature)
        self.fcs = nn.ModuleList([nn.Linear(hidden_feature, hidden_feature) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_feature) for _ in range(layers)])
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.bn(x)
        x = self.relu(x)
        for b, l in zip(self.bns, self.fcs):
            x = l(x)
            x = b(x)
            x = self.relu(x)
            x = self.do(x)
        x = self.relu(self.fc_out(x))
        x = self.do(x)
        x = self.sm(x)
        return x


def train(cv_idx):
    device = "cuda"

    train_data = dataset.CV_SAP_data_set("./data/2nd", cv_index=cv_idx)
    test_data = dataset.CV_SAP_data_set("./data/2nd", cv_index=cv_idx, train=False)
    data_len = len(train_data)

    model = net(238, 2000, 21, layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    cri = torch.nn.MSELoss()

    epochs = 500
    report_every = 2
    batch_size = 64
    last = (0, 0)
    last_loss = None
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
        print("CV {} \t[Test] Epoch {:3d} : loss {:.6f} [ {} / {} ( {:.2f} % )]".format(cv_idx, epoch + 1,
                                                                                        bloss / test_len, correct,
                                                                                        test_len,
                                                                                        float(float(correct) / float(
                                                                                            test_len)) * 100))
        if last_loss is None or correct >= last[0]:
            last = (correct, test_len)
            last_loss = bloss
    print()
    return last, last_loss


def main():
    score = [0, 0]
    loss = 0.0
    for cv_idx in range(10):
        (a, b), c = train(cv_idx)
        score[0] += a
        score[1] += b
        loss += c
    print("[Final] : Loss : {}\t {} / {} ( {:.2f} % )".format(loss / float(score[1]), *score, (float(score[0]) / float(score[1])) * 100))


if __name__ == "__main__":
    main()
