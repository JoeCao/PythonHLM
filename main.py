# coding=utf-8
from __future__ import unicode_literals

import gensim
import jieba
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch


def find_main_charecters(num=10):
    with open('hlm.txt') as f:
        data = f.read()
    with open('names.txt') as f:
        characters_names = [line.strip('\n') for line in f.readlines()]
    count = []
    for name in characters_names:
        count.append([name, data.count(name)])
    count.sort(key=lambda x: x[1])
    _, ax = plt.subplots()

    numbers = [x[1] for x in count[-num:]]
    names = [x[0] for x in count[-num:]]
    ax.barh(range(num), numbers, color='red', align='center')
    ax.set_title('红楼梦')
    ax.set_yticks(range(num))
    ax.set_yticklabels(names)
    plt.show()


def find_all_names():
    with open('names.txt') as f:
        characters_names = [line.strip('\n') for line in f.readlines()]
    return characters_names


def add_to_dict(characters_names):
    for name in characters_names:
        jieba.add_word(name)


def tranning():
    with open('hlm.txt') as f:
        data = [line.strip()
                for line in f.readlines()
                if line.strip()]

    sentences = []
    for line in data:
        words = list(jieba.cut(line))
        sentences.append(words)
    model = gensim.models.Word2Vec(sentences,
                                   size=100,
                                   window=5,
                                   min_count=5,
                                   workers=4)
    return model


def find_relationship(tranning_model, a, b, c):
    """
    返回 d
    a与b的关系，跟c与d的关系一样
    """
    d, _ = tranning_model.most_similar(positive=[c, b], negative=[a])[0]
    print("给定“{}”与“{}”，“{}”和“{}”有类似的关系".format(a, b, c, d))


def kmeans(model, characters_names):
    all_names = []

    word_vectors = None
    np_names = None
    for name in characters_names:
        if name in model:
            all_names.append(name)
    for name in all_names:
        if word_vectors is None:
            word_vectors = model[name]
        else:
            word_vectors = np.vstack((word_vectors, model[name]))
            np_names = np.array(all_names)

    return np_names, word_vectors


def aggre3(np_names, word_vectors):
    N = 3

    label = KMeans(N).fit(word_vectors).labels_

    for c in range(N):
        print("类别{}：".format(c + 1))
        for idx, name in enumerate(np_names[label == c]):
            print(name, end=", ")
            if idx % 10 == 9:
                print('')
        print('')


def aggre4(np_names, word_vectors):
    N = 4
    label = KMeans(N).fit(word_vectors).labels_
    c = sp.stats.mode(label).mode

    remain_names = np_names[label != c]
    remain_vectors = word_vectors[label != c]
    remain_label = KMeans(N).fit(remain_vectors).labels_

    for c in range(N):
        print("类别{}：".format(c + 1))
        for idx, name in enumerate(remain_names[remain_label == c]):
            print(name, end=", ")
            if idx % 10 == 9:
                print('')
        print('')


def hierarchy(np_names, word_vectors):
    font_yahei_consolas = FontProperties(fname="/Library/Fonts/华文仿宋.ttf")
    Y = sch.linkage(word_vectors, method="ward")

    _, ax = plt.subplots(figsize=(10, 40))

    Z = sch.dendrogram(Y, orientation='right')
    idx = Z['leaves']

    ax.set_xticks([])
    ax.set_yticklabels(np_names[idx], fontproperties=font_yahei_consolas,
                       fontsize=14)
    ax.set_frame_on(False)

    plt.show()


if __name__ == "__main__":
    # find_main_charecters(10)
    add_to_dict(find_all_names())
    model = tranning()
    for k, s in model.most_similar(positive=["宝钗"]):
        print('{}:{}'.format(k, s))
    find_relationship(model, '王夫人', '凤姐', '薛姨妈')
    # np_names, word_vectors = kmeans(model, find_all_names())
    # aggre4(np_names, word_vectors)
    # hierarchy(np_names, word_vectors)
