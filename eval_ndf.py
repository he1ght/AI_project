import argparse
import logging

import torch
import torch.nn.functional as F
import torchvision

import dataset
import ndf

CV_FOLD = 10


def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='eval_ndf.py')
    parser.add_argument('-dataset', choices=['mnist', '1st', '2nd'], default='1st')
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-gpuid', type=int, default=0)

    parser.add_argument('-report_every', type=int, default=1)
    parser.add_argument('-model', type=str, default="./model/model.1st.final.pt")

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
        re_dataset = dataset.CV_SAP_data_set('./data/1st', cv_index=opt.cv_index, train=False)
        return re_dataset
    elif opt.dataset == '2nd':
        re_dataset = dataset.CV_SAP_data_set('./data/2nd', cv_index=opt.cv_index, train=False)
        return re_dataset
    else:
        raise NotImplementedError


def prepare_model(opt):
    try:
        checkpoint = torch.load(opt.model)
    except FileNotFoundError:
        checkpoint = torch.load(opt.model + ".cv{}.pt".format(opt.cv_index))
    model_opt = checkpoint['opt']
    if opt.dataset == 'mnist':
        feat_layer = ndf.MNISTFeatureLayer(model_opt.feat_dropout, shallow=True)
    elif opt.dataset == '1st':
        feat_layer = ndf.UCIStudPerformLayer(model_opt.feat_dropout, model_opt.hidden_size)
    elif opt.dataset == '2nd':
        feat_layer = ndf.UCIStudLayer(model_opt.feat_dropout, model_opt.hidden_size)
    else:
        raise NotImplementedError

    forest = ndf.Forest(n_tree=model_opt.n_tree, tree_depth=model_opt.tree_depth,
                        n_in_feature=feat_layer.get_out_feature_size(),
                        tree_feature_rate=model_opt.tree_feature_rate, n_class=model_opt.n_class,
                        dropout_rate=model_opt.feat_dropout)
    model = ndf.NeuralDecisionForest(feat_layer, forest)
    model.load_state_dict(checkpoint['model'])

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def eval(opt):
    # Eval

    test_loss = 0
    correct = 0
    cnt = 0
    test_len = 0
    for i in range(CV_FOLD):

        opt.cv_index = i
        model = prepare_model(opt)
        model.eval()
        data_set = prepare_data(opt)
        test_loader = torch.utils.data.DataLoader(data_set, batch_size=opt.batch_size, shuffle=True)

        print("=-=-=-=-=-=-=-=  Model {} =-=-=-=-=-=-=-=-=".format(i + 1))
        with torch.no_grad():
            for data, target in test_loader:
                if opt.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.nll_loss(torch.log(output), target, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                for pr, tg in zip(pred, target):
                    cnt += 1
                    print("=== {} data ==================================".format(cnt))
                    if opt.dataset == '1st':
                        print("predict : {} \n target : {}".format(data_set.fields_y['i2c'][pr.item()],
                                                               data_set.fields_y['i2c'][tg.item()]))
                    else:
                        print("predict : {} \n target : {}".format(data_set.fields_y[-1]['i2c'][pr.item()],
                                                                   data_set.fields_y[-1]['i2c'][tg.item()]))
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_len += len(test_loader.dataset)
    test_loss /= test_len
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f} %)\n'.format(
        test_loss, correct, test_len,
        float(float(correct) / float(test_len)) * 100))


def main():
    opt = parse_arg()

    opt.cuda = opt.gpuid >= 0
    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
        opt.device = 'cuda:{}'.format(opt.cuda)
    else:
        opt.device = 'cpu'

    eval(opt)


if __name__ == '__main__':
    main()
