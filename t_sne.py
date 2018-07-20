# class prepare(self,filename):
#     self.filename = filename

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from time import time


def prepare_y(resultDoc):
    result = []
    with open(resultDoc) as fin:
        node_num, size = [int(x) for x in fin.readline().strip().split()]
        while True:
            lineResult = fin.readline().strip()
            if not len(lineResult):
                break
            result.append(int(lineResult))
    return np.array(result)

def prepare_x(vecDoc):
    vec = {}
    with open(vecDoc) as fin:
        node_num, size = [int(x) for x in fin.readline().strip().split()]
        while True:
            pre_vec = fin.readline().strip().split()
            if not len(pre_vec):
                break
            vec[int(pre_vec[0])] =  pre_vec[1:]
    return DataFrame(vec).T
        

def getTsne(emb_vec,emb_result):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(emb_vec)
    y = emb_result
    plot_embedding(X_tsne, y,
                "t-SNE embedding of the digits (time %.2fs)" %
                (time() - t0))
    plt.savefig('t_sne.png')
    plt.show()  


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],
                 color=plt.cm.Set1(y[i] / 10.))
    if title is not None:
        plt.title(title)

def pca(X, n_components):
    pca = PCA(n_components)
    pca.fit(X)
    return pca.transform(X)


getTsne(prepare_x("vec_all.txt"),prepare_y("result.txt"))




