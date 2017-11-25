import numpy as np
import pandas as pd


def load_data():
    feature = [];
    label = []
    with open('/Users/hanzhao/PycharmProjects/MLstudy/file/lrdataset.txt') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            feature.append([1.0, float(line_arr[0]), float(line_arr[1])])  # lr input dataset with (1,x0,x1...xn)T
            label.append(int(line_arr[2]))
    return feature, label


def load_datas():
    csv_data = pd.read_csv("/Users/hanzhao/PycharmProjects/MLstudy/file/lrdataset.", delim_whitespace=True, header=0)
    data = np.array(csv_data)
    data_tranpose = np.transpose(data)
    feature = data_tranpose[:2]
    label = data_tranpose[2:]
    return feature, label
