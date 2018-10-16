# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

'''@descripting 定义一个函数来对n折的训练集和测试集进行预测,此函数返回每个模型对训练集和测试集的预测'''


def Stacking(model, train, y, test, n_fold):
    folds = StratifiedKFold(n_splits=n_fold, random_state=1)
    test_pred = np.empty((test.shape[0], 1), float)
    train_pred = np.empty((0, 1), float)

    for train_indices, val_indices in folds.split(train, y.values):
        x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        model.fit(X=x_train, y=y_train)
        train_pred = np.append(train_pred, model.predict(x_val))
        test_pred = np.append(test_pred, model.predict(test))
    return test_pred.reshape(-1, 1), train_pred


'''第一层stacking'''


def stack_model_first(x_train, x_test, y_train):
    model1 = tree.DecisionTreeClassifier(random_state=1)
    test_pred1, train_pred1 = Stacking(model=model1, n_fold=10, train=x_train, test=x_test, y=y_train)
    train_pred1 = pd.DataFrame(train_pred1)
    test_pred1 = pd.DataFrame(test_pred1)
    return train_pred1, test_pred1


'''第二层stacking'''


def stack_model_second(x_train, x_test, y_train):
    model2 = KNeighborsClassifier()
    test_pred2, train_pred2 = Stacking(model=model2, n_fold=10, train=x_train, test=x_test, y=y_train)

    train_pred2 = pd.DataFrame(train_pred2)
    test_pred2 = pd.DataFrame(test_pred2)
    return train_pred2, test_pred2


'''第一层，第二层stacking基层'''


def stack_model_base(train_pred1, train_pred2, test_pred1, test_pred2, y_train, y_test):
    df = pd.concat([train_pred1, train_pred2], axis=1)
    df_test = pd.concat([test_pred1, test_pred2], axis=1)

    model = LogisticRegression(random_state=1)
    model.fit(df, y_train)
    model.score(df_test, y_test)


if __name__ == '__main__':
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_pred1, test_pred1 = stack_model_first(x_train, x_test, y_train)
    train_pred2, test_pred2 = stack_model_second(x_train, x_test, y_train)
    stack_model_base(train_pred1, train_pred2, test_pred1, test_pred2, y_train, y_test)
