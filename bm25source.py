# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module contains function of computing rank scores for documents in
corpus and helper class `BM25` used in calculations. Original algorithm
descibed in [1]_, also you may check Wikipedia page [2]_.
.. [1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond,
       http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
.. [2] Okapi BM25 on Wikipedia, https://en.wikipedia.org/wiki/Okapi_BM25
"""

import math
from six import iteritems
from six.moves import xrange

PARAM_K1 = 1.5
PARAM_K2 = 1.2
PARAM_B = 0.75
EPSILON = 0.25
"""
k1的作用是对查询词在文档中的词频进行调节，
如果将 k1设定为 0，则第二部分计算因子成了整数 1，即不考虑词频的因素，退化成了二元独立模型。 
如果将 k1设定为较大值， 则第二部分计算因子基本和词频 fi保持线性增长，即放大了词频的权值。
根据经验，一般将 k1设定为 1.2。

调节因子 k2和 k1的作用类似，不同点在于其是针对查询词中的词频进行调节，
一般将这个值设定在 0 到 1000 较大的范围内。
之所以如此，是因为查询往往很短，所以不同查询词的词频都很小，
词频之间差异不大，较大的调节参数数值设定范围允许对这种差异进行放大。

b是调节因子，极端情况下，将 b 设定为 0，则文档长度因素将不起作用，
经验表明一般将 b 设定为 0．75 会获得较好的搜索效果。 
"""


class BM25(object):
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        q_count = {}
        for i in document:
            if i in q_count:
                q_count[i] += 1.0
            else:
                q_count[i] = 1.0

        score = 0
        for w in set(document):
            if w not in self.f[index]:
                continue
            idf = self.idf[w] if self.idf[w] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][w]) \
                     * ((PARAM_K1 + 1) / (
                self.f[index][w] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avgdl))) \
                     * (((PARAM_K2 + 1.0) * q_count[w]) / (PARAM_K2 + q_count[w]))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


def get_bm25_weights(corpus):
    bm25 = BM25(corpus)
    average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

    weights = []
    for doc in corpus:
        scores = bm25.get_scores(doc, average_idf)
        weights.append(scores)

    return weights
