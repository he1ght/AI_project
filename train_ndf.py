import argparse
import logging

import os
import torch
import torch.nn.functional as F
import torchvision
from datetime import datetime

import dataset
import ndf

CV_FOLD = 10


def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='train_ndf.py')
    parser.add_argument('-dataset', choices=['mnist', '1st', '2nd'], default='mnist')
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-feat_dropout', type=float, default=0.3)

    parser.add_argument('-n_tree', type=int, default=10)
    parser.add_argument('-tree_depth', type=int, default=5)
    parser.add_argument('-n_class', type=int, default=3)
    parser.add_argument('-tree_feature_rate', type=float, default=0.5)
    parser.add_argument('-hidden_size', type=int, default=256)
    parser.add_argument('-tree_iter', type=int, default=5)

    parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-jointly_training', action='store_true', default=False)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-cv_index', type=int, default=0)

    parser.add_argument('-report_every', type=int, default=1)
    parser.add_argument('-save', type=str, default="model/test")
    parser.add_argument('-log', action='store_true', default=False)
    parser.add_argument('-log_dir', type=str, default="log/")
    parser.add_argument('-only_best', action='store_true', default=False)

    opt = parser.parse_args()
    return opt


def prepare_data(opt):
    if opt.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                   ]))

        eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                  ]))
        return {'train': train_dataset, 'eval': eval_dataset}
    if opt.dataset == '1st':
        train_dataset = dataset.CV_SAP_data_set('./data/1st', cv_index=opt.cv_index, train=True)
        eval_dataset = dataset.CV_SAP_data_set('./data/1st', cv_index=opt.cv_index, train=False)
        return {'train': train_dataset, 'eval': eval_dataset}
    elif opt.dataset == '2nd':
        train_dataset = dataset.CV_SAP_data_set('./data/2nd', cv_index=opt.cv_index, train=True)
        eval_dataset = dataset.CV_SAP_data_set('./data/2nd', cv_index=opt.cv_index, train=False)
        return {'train': train_dataset, 'eval': eval_dataset}
    else:
        raise NotImplementedError


def prepare_model(opt):
    if opt.dataset == 'mnist':
        feat_layer = ndf.MNISTFeatureLayer(opt.feat_dropout, shallow=True)
    elif opt.dataset == '1st':
        feat_layer = ndf.UCIStudPerformLayer(opt.feat_dropout, opt.hidden_size)
    elif opt.dataset == '2nd':
        feat_layer = ndf.UCIStudLayer(opt.feat_dropout, opt.hidden_size)
    else:
        raise NotImplementedError
    forest = ndf.Forest(n_tree=opt.n_tree, tree_depth=opt.tree_depth, n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=opt.tree_feature_rate, n_class=opt.n_class,
                        dropout_rate=opt.feat_dropout)
    model = ndf.NeuralDecisionForest(feat_layer, forest)

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def prepare_optim(model, opt):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-3)


def train(model, optim, db, opt, writer=None):
    last_score = (0, 0)
    last_loss = 0.0
    log_softmax = torch.nn.LogSoftmax(dim=1)
    for epoch in range(1, opt.epochs + 1):
        # Compute pi by iterating
        cls_onehot = torch.eye(opt.n_class)
        feat_batches = []
        target_batches = []
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)

        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target, cls_onehot = data.cuda(), target.cuda(), cls_onehot.cuda()
            # Get feats
            with torch.no_grad():
                feats = model.feature_layer(data)
                feats = feats.view(feats.size(0), -1)
                feat_batches.append(feats)
            target_batches.append(cls_onehot[target])
        with torch.no_grad():
            for tree_idx, tree in enumerate(model.forest.trees):
                mu_batches = []
                for feats in feat_batches:
                    mu = tree(feats)                                                 # [batch_size,n_leaf]
                    mu_batches.append(mu)
                for _ in range(opt.tree_iter):
                    new_pi = torch.zeros((tree.n_leaf, tree.n_class)).to(opt.device) # [n_leaf,n_class]
                    for mu, target in zip(mu_batches, target_batches):
                        pi = tree.get_pi()                                           # [n_leaf,n_class]
                        prob = tree.calculate_probability(mu, pi)                    # [batch_size,n_class]

                        _target = target.unsqueeze(1)                                # [batch_size,1,n_class]
                        _pi = pi.unsqueeze(0)                                        # [1,n_leaf,n_class]
                        _mu = mu.unsqueeze(2)                                        # [batch_size,n_leaf,1]
                        _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)     # [batch_size,1,n_class]
                        _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob    # [batch_size,n_leaf,n_class]
                        new_pi += torch.sum(_new_pi, dim=0)
                    new_pi = F.softmax(new_pi, dim=1).data
                    tree.update_pi(new_pi)

        # Training theta
        model.train()
        bloss = 0
        train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            optim.zero_grad()
            output = model(data)
            output = torch.log(output)
            loss = F.nll_loss(output, target)
            loss.backward()
            bloss += loss.item()
            optim.step()
        bloss /= len(train_loader)

        # Training log
        # print('{} CV\tTrain Epoch: {}\tLoss: {:.6f}'.format(opt.cv_index, epoch, bloss))
        if writer is not None:
            writer.add_scalar('train/model_cv%d' % opt.cv_index, bloss, epoch)

        # Eval
        model.eval()
        test_loss = 0
        correct = 0
        test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=True)
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                output = torch.log(output)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_len = len(test_loader.dataset)
        print('[TEST] CV [{}/{}]\tEpoch [{}/{}]\tAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f} %)'.format(opt.cv_index,
                                                                                                          CV_FOLD - 1,
                                                                                                          epoch,
                                                                                                          opt.epochs,
                                                                                                          test_loss / test_len,
                                                                                                          correct,
                                                                                                          test_len,
                                                                                                          float(float(
                                                                                                              correct) / float(
                                                                                                              test_len)) * 100))

        if writer is not None:
            writer.add_scalar('test/model_cv%d' % opt.cv_index, test_loss, epoch)
        if not opt.only_best or correct >= last_score[0]:
            last_score = (correct, test_len)
            last_loss = test_loss
            save_data = {
                'model': model.state_dict(),
                'opt': opt
            }
            torch.save(save_data, opt.model_dir + opt.save.split("/")[-1] + ".cv{}.pt".format(opt.cv_index))
    print()
    return last_score, last_loss


def merge_models(opt):
    real_model = prepare_model(opt)
    for p in real_model.parameters():
        p.data.fill_(0.0)
    for i in range(CV_FOLD):
        model = prepare_model(opt)
        checkpoint = torch.load(opt.save + ".cv{}.pt".format(i))
        model.load_state_dict(checkpoint['model'])
        for rp, p in zip(real_model.parameters(), model.parameters()):
            rp.data.add_(p.data)
    # for p in real_model.parameters():
    #     p.data.div_(10)
    return real_model


def main():
    opt = parse_arg()

    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
        opt.device = 'cuda:{}'.format(opt.gpuid)
    else:
        opt.device = 'cpu'
    opt.time = datetime.now().strftime("%b-%d_%H-%M-%S")
    opt.model_dir = "/".join(opt.save.split("/")[:-1]) + "/" + opt.time + "/"

    if opt.log:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(
            opt.log_dir + opt.time + "_{}".format(opt.dataset))
    else:
        writer = None
    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)
    score = [0, 0]
    loss = 0.0
    for i in range(CV_FOLD):
        opt.cv_index = i
        data_set = prepare_data(opt)
        model = prepare_model(opt)
        optim = prepare_optim(model, opt)
        s, l = train(model, optim, data_set, opt, writer)
        score[0] += s[0]
        score[1] += s[1]
        loss += l

    if writer is not None:
        writer.close()
    print("\n" + "*" * 40)
    print("* Loss:{} \t {} / {} [ {:.2f} % ]".format(loss/score[1], *score, (float(score[0]) / float(score[1])) * 100))


if __name__ == '__main__':
    main()
