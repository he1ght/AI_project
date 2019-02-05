import os
import pickle
from operator import eq
import argparse
import logging
import numpy as np
import torch
from torchvision.transforms import transforms
import pandas as pd

CV_FOLD = 10


def parse_arg():
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s]: %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description='preprocess.py')
    parser.add_argument('-dataset', choices=['1st', '2nd'], default='1st')
    parser.add_argument('-data', type=str, default="./data/1st")
    parser.add_argument('-save', type=str, default="./data/1st")
    parser.add_argument('-shuffle', action='store_true', default=False)

    opt = parser.parse_args()
    return opt


def save_obj(obj, name, path="."):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path="."):
    with open(path + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def one_hot_embedding(labels, num_classes):
    y = np.eye(num_classes)
    return y[labels].tolist()


class Preprocess_1st:
    def __init__(self, root, out, k_fold=10, shuffle=True):
        self.root = os.path.expanduser(root)
        self.out = out

        self.k_fold = k_fold

        self.fields_x = []
        self.fields_y = {}
        self.X = []
        self.y = []

        is_data = False
        file = os.path.join(self.root, "Sapfile1.arff")
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            if not line:
                continue
            if is_data:
                ct_data = line.split(",")
                self.X.append(ct_data)
                self.y.append(self.X[-1].pop())
            else:
                etr = line.split(" ")
                attr = etr.pop(0)
                if eq(attr, "@ATTRIBUTE"):
                    etr[-1] = etr[-1].replace("{", "").replace("}", "")
                    ldict = {'type': 'cate', 'name': etr.pop(0), 'choices': etr.pop().split(',')}
                    ldict['len'] = len(ldict['choices'])
                    self.fields_x.append(ldict)

                elif eq(attr, "@DATA"):
                    is_data = True
        for field in self.fields_x:
            field['c2i'] = dict([(w, i) for i, w in enumerate(field['choices'])])
            field['i2c'] = dict([(i, w) for i, w in enumerate(field['choices'])])
        self.fields_y = self.fields_x.pop()
        new_x = []
        new_y = []
        for d in self.X:
            # new_x.append([self.fields_x[i]['c2i'][e] for i, e in enumerate(d)])
            # print([one_hot_embedding(self.fields_x[i]['c2i'][e], self.fields_x[i]['len']) for i, e in enumerate(d)])
            temp_vec = [one_hot_embedding(self.fields_x[i]['c2i'][e], self.fields_x[i]['len']) for i, e in enumerate(d)]
            new_vec = []
            for v in temp_vec:
                for vv in v:
                    new_vec.append(vv)
            new_x.append(new_vec)
        for i, e in enumerate(self.y):
            new_y.append(self.fields_y['c2i'][e])
        if shuffle:
            import random
            pairs = list(zip(new_x, new_y))
            random.shuffle(pairs)
            new_x, new_y = zip(*pairs)

        self.X = torch.Tensor(new_x)
        self.y = torch.LongTensor(new_y)

    def preprocess(self):
        self.save_fields()
        self.save_cv_data()

    def save_fields(self):
        save_obj(self.fields_x, "fields", path=self.out)
        save_obj(self.fields_y, "atd", path=self.out)

    def save_cv_data(self):
        cv_range = np.linspace(0, len(self.X), self.k_fold + 1, dtype=np.int)
        for i in range(len(cv_range)):
            if i + 1 == len(cv_range):
                break
            torch.save(self.X[cv_range[i]:cv_range[i + 1]], self.out + "/data.{}.pt".format(i))
            torch.save(self.y[cv_range[i]:cv_range[i + 1]], self.out + "/result.{}.pt".format(i))


class Preprocess_2nd:
    def __init__(self, root, out, k_fold=10, shuffle=False):
        self.root = os.path.expanduser(root)
        self.out = out

        self.k_fold = k_fold

        self.X = []
        self.y = []

        # fea = []
        # with open(os.path.join(self.root, "student_dict.txt"), encoding="utf-8") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         data = line.replace("\n", "").split(" ")
        #         data.pop(0)
        #         name = data.pop(0)
        #         dtype = data.pop(0)
        #         if dtype == 'num':
        #             data[0] = int(data[0])
        #             data[1] = int(data[1])
        #             length = data[1] - data[0] + 1
        #             a, b = data
        #             data = []
        #             for i in range(a, b + 1):
        #                 data.append(str(i))
        #         elif dtype == 'exp':
        #             data = ['1', '2', '4']
        #             length = 3
        #         elif dtype == 'cate':
        #             length = len(data)
        #         fea.append({'name':name,'type':dtype,'choices':data,'len':length})
        #         fea[-1]['c2i'] = dict([(w, i) for i, w in enumerate(fea[-1]['choices'])])
        #         fea[-1]['i2c'] = dict([(i, w) for i, w in enumerate(fea[-1]['choices'])])
        # print(fea)

        self.fields_x = [
            {'name': 'school', 'type': 'cate', 'choices': ['"GP"', '"MS"'], 'len': 2},
            {'name': 'sex', 'type': 'cate', 'choices': ['"F"', '"M"'], 'len': 2},
            {'name': 'age', 'type': 'num', 'choices': ['15', '16', '17', '18', '19', '20', '21', '22'], 'len': 8},
            {'name': 'address', 'type': 'cate', 'choices': ['"U"', '"R"'], 'len': 2},
            {'name': 'famsize', 'type': 'cate', 'choices': ['"LE3"', '"GT3"'], 'len': 2},
            {'name': 'Pstatus', 'type': 'cate', 'choices': ['"T"', '"A"'], 'len': 2},
            {'name': 'Medu', 'type': 'num', 'choices': ['0', '1', '2', '3', '4'], 'len': 5},
            {'name': 'Fedu', 'type': 'num', 'choices': ['0', '1', '2', '3', '4'], 'len': 5},
            {'name': 'Mjob', 'type': 'cate', 'choices': ['"teacher"', '"health"', '"services"', '"at_home"', '"other"'],
             'len': 5},
            {'name': 'Fjob', 'type': 'cate', 'choices': ['"teacher"', '"health"', '"services"', '"at_home"', '"other"'],
             'len': 5},
            {'name': 'reason', 'type': 'cate', 'choices': ['"home"', '"reputation"', '"course"', '"other"'], 'len': 4},
            {'name': 'guardian', 'type': 'cate', 'choices': ['"mother"', '"father"', '"other"'], 'len': 3},
            {'name': 'traveltime', 'type': 'num', 'choices': ['1', '2', '3', '4'], 'len': 4},
            {'name': 'studytime', 'type': 'num', 'choices': ['1', '2', '3', '4'], 'len': 4},
            {'name': 'failures', 'type': 'exp', 'choices': ['1', '2', '4'], 'len': 3},
            {'name': 'schoolsup', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'famsup', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'paid', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'activities', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'nursery', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'higher', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'internet', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'romantic', 'type': 'cate', 'choices': ['"yes"', '"no"'], 'len': 2},
            {'name': 'famrel', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'freetime', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'goout', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'Dalc', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'Walc', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'health', 'type': 'num', 'choices': ['1', '2', '3', '4', '5'], 'len': 5},
            {'name': 'absences', 'type': 'num',
             'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                         '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48',
                         '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64',
                         '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80',
                         '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93'], 'len': 94},

            {'name': 'G1', 'type': 'num',
             'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20'], 'len': 21},
            {'name': 'G2', 'type': 'num',
             'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20'], 'len': 21},
        ]
        self.fields_y = [
            # {'name': 'G1', 'type': 'num',
            #  'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
            #              '17', '18', '19', '20'], 'len': 21},
            # {'name': 'G2', 'type': 'num',
            #  'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
            #              '17', '18', '19', '20'], 'len': 21},
            {'name': 'G3', 'type': 'num',
             'choices': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                         '17', '18', '19', '20'], 'len': 21}]
        lambda_exception = lambda n: n if 1 <= n < 3 else 4
        for field in self.fields_x:
            field['c2i'] = dict([(w, i) for i, w in enumerate(field['choices'])])
            field['i2c'] = dict([(i, w) for i, w in enumerate(field['choices'])])
        for field in self.fields_y:
            field['c2i'] = dict([(w, i) for i, w in enumerate(field['choices'])])
            field['i2c'] = dict([(i, w) for i, w in enumerate(field['choices'])])

        # pd.DataFrame(os.path.join(self.root, "student-mat.csv"),sep=";",header=True)
        df1 = pd.read_csv(os.path.join(self.root, "student-mat.csv"), sep=";", header=0)
        df2 = pd.read_csv(os.path.join(self.root, "student-por.csv"), sep=";", header=0)
        df3 = np.concatenate((df1.values, df2.values))
        df3 = df3.tolist()
        for data in df3:
            vec_x = []
            for i, field in enumerate(self.fields_x):
                d = data.pop(0)
                if field['type'] == 'exp':
                    d = lambda_exception(d)
                if type(d) == int:
                    d = str(d)
                else:
                    d = '"{}"'.format(d)
                vec_x += one_hot_embedding(field['c2i'][d], field['len'])
            self.X.append(vec_x)

            for i, field in enumerate(self.fields_y):
                d = data.pop(0)
                d = str(d)
                vec_y = field['c2i'][d] # only final degree
            self.y.append(vec_y)
        if shuffle:
            import random
            pairs = list(zip(self.X, self.y))
            random.shuffle(pairs)
            self.X, self.y = zip(*pairs)

        self.X = torch.Tensor(self.X)
        self.y = torch.LongTensor(self.y)

    def preprocess(self):
        self.save_fields()
        self.save_cv_data()

    def save_fields(self):
        save_obj(self.fields_x, "fields", path=self.out)
        save_obj(self.fields_y, "atd", path=self.out)

    def save_cv_data(self):
        cv_range = np.linspace(0, len(self.X), self.k_fold + 1, dtype=np.int)
        for i in range(len(cv_range)):
            if i + 1 == len(cv_range):
                break
            torch.save(self.X[cv_range[i]:cv_range[i + 1]], self.out + "/data.{}.pt".format(i))
            torch.save(self.y[cv_range[i]:cv_range[i + 1]], self.out + "/result.{}.pt".format(i))


if __name__ == "__main__":
    opt = parse_arg()
    if opt.dataset == "1st":
        s = Preprocess_1st(opt.data, k_fold=CV_FOLD, out=opt.save, shuffle=opt.shuffle)
    elif opt.dataset == "2nd":
        s = Preprocess_2nd(opt.data, k_fold=CV_FOLD, out=opt.save, shuffle=opt.shuffle)
    s.preprocess()
