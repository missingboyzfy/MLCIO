import numpy as np
import pandas as pd
import random

from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors


def get_minority_instance(X, y):
    """
        获取到少数样本
    """
    index = get_index(y)  # index就是少数类中值为1的索引的列表
    X_sub = X[X.index.isin(index)].reset_index(drop=True)  # 做数据清洗，选出少数样本(特征向量)，并将其索引重新排序
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub  # 返回的都是DataFrame形式


def get_index(df):
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)  # 获取到['class_1', 'class_3']值为1的索引(也就是行)
        index = index.union(sub_index)
    return list(index)


def get_tail_label(df):
    """
        尾标签就是少数标签，头标签就是大多数标签
    """
    columns = df.columns  # Index(['class_0', 'class_1', 'class_2', 'class_3', 'class_4'], dtype='object')
    n = len(columns)  # 5
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]  # columns[column]表示列名,统计每一列中1的个数并保存到irpl中
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    # print(tail_label)
    return tail_label  # 返回['class_1', 'class_3']


def nearest_neighbour(X):  # 获取少数样本的最近邻样本
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample, m):
    """
        X:少数样本的特征集，DataFrame形式
        y:少数样本的标签集，DataFrame形式
        n_sample:要合成的样本数
    """
    indices2 = nearest_neighbour(X)  # 获取少数样本的邻居索引，以二维数组的形式返回。比如[[0 13 6 12 8], [1 28 13 29 8]...]
    n = len(X)  # 33，即少数样本数
    new_X = np.zeros((n_sample, X.shape[1]))  # X.shape[1]代表10列，二维数组 [[], [],,,[]]， new_X用于存放新样本的特征集
    target = np.zeros((n_sample, y.shape[1]))  # y.shape[1]代表5列,二维数组[[], [],,,[]]，target用于存放新样本的标签集
    # print(target.shape)
    ser = y.sum(axis=0, skipna=True)  # 获得少数标签中样本的数量
    ser_arr = np.array(ser)  # 转为数组形式,会随着标签的数量变化而变化

    # 获得当前种子样本和其K近邻样本组成集合的标准差
    # std_value = np.array(X.describe().loc["std", :])

    np.seterr(invalid='ignore')  # 忽略分母为0的运算
    for i in range(n_sample):
        # 1.合成特征集，结合切比雪夫定理
        # 随机选取一个少数样本，以其为均值点
        reference_index = random.randint(0, n - 1)
        reference_featureSet = np.array(X.loc[reference_index, :])
        # 获得当前种子样本和其K近邻样本组成集合的标准差
        all_point = indices2[reference_index]
        ref_Neighbours = X[X.index.isin(all_point)]
        std_value = np.array(ref_Neighbours.describe().loc["std", :])

        std_value = [each * random.choice([-1, 1]) for each in std_value]
        new_featureSet = np.add(np.array(reference_featureSet), np.array(std_value) * m)
        new_X[i] = new_featureSet

                # 1.获取到所选少数样本及其邻居样本的少数标签
        all_point = indices2[reference_index]  # [25 17 13 18 32]
        nn_df = y[y.index.isin(all_point)]
        minor_labels = []  # ["class_1"], ["class_1", "class_3"]
        for j in nn_df.columns:
            col_arr = np.array(nn_df[j])
            if 1 in col_arr:
                minor_labels.append(j)

        # 2.计算每个少数标签下的样本总数n0, 见循环上方  即ser_arr
        # 3.计算当前种子样本和邻居样本中少数标签下的少数样本数 m0
        ser_one = nn_df.sum(axis=0, skipna=True)
        ser_one_arr = np.array(ser_one)  # [0 1 0 4 0]  [0 0 0 5 0]  [0 5 0 0 0]...
        # print(ser_one_arr)
        # 4.计算权重
        weight = []

        for j in range(len(ser_one_arr)):
            weight.append(ser_one_arr[j] / ser_arr[j])
        # print(weight)

        # 进行排序赋权，即找到当前标签下1的个数乘以对应权重，总权大的定位合成样本的少数标签
        target_zero = np.zeros(y.shape[1])
        target_zero_arr = weight * ser_one_arr
        target_zero_arr = np.array(target_zero_arr)
        indices_argsort = np.argsort(target_zero_arr)  # indices_argsort保存的是权值数组排序后的原数组的索引。接下来我只要把新样本后y.shape[1] / 2的标签赋值为1即可。如果
                                                                # 将前面几位赋值1的话，岂不是类似的标签更加的浓密。。。
        # print(indices_argsort)
        num = int(y.shape[1] / 2)  # 加不加1是个问题呢
        # print(indices_argsort[-num:])
        indices_argsort_arr = indices_argsort[-num:]
        # print(indices_argsort_arr)
        for k in indices_argsort_arr:
            target_zero[k] = 1
        # print(target_zero)
        target[i] = target_zero


    # print(target)
    new_X = pd.DataFrame(new_X, columns=X.columns)  # 转成DataFrame形式,此乃合成样本的特征集，100个
    target = pd.DataFrame(target, columns=y.columns)
    new_X_concat = pd.concat([X, new_X], axis=0).reset_index(drop=True)  # 并到原先的33个少数样本中
    target_concat = pd.concat([y, target], axis=0)
    return new_X_concat, target_concat


if __name__ == '__main__':
    """
     本算法的主函数
    """
    # X, y = create_dataset()  # 创建一个数据集，Dataframe形式
    X, y = datasets.make_multilabel_classification(n_samples=1000, n_features=10, n_classes=7, n_labels=2, length=50)
    cl = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6"]
    y = pd.DataFrame(y, columns=cl)
    X = pd.DataFrame(X)
    X_sub, y_sub = get_minority_instance(X, y)  # 获得少数样本
    n_sample = 0.2 * X.shape[0]
    X_res, y_res = MLSMOTE(X_sub, y_sub, int(n_sample), 1)  # 做数据增强 ，即合成样本，包括特征集和标签集
    # print(pd.concat([X, X_res], ignore_index=True))
    # print(y_res.shape)
    print(pd.concat([X, X_res], axis=0).reset_index(drop=True))
    print(pd.concat([y, y_res], axis=0).reset_index(drop=True))




