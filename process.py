# coding=utf-8
import re
import jieba.posseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import codecs
from numpy import linalg as la
from gensim import corpora
from bm25source import BM25


# from gensim.summarization import bm25
def quchong(contents):
    """
    :param contents: 
    :return:old_ids:the original id of the quchong_content.new_content
    """
    old_ids = []
    new_contents = []
    all_content = set()
    for i, content in enumerate(contents):
        if i == 0:
            continue
        content = content.split(',')[1]
        new_content = preforquchong(content)
        if new_content not in all_content:
            all_content.add(new_content)
            old_ids.append(i)
            new_contents.append(content)
    return old_ids, new_contents


def preforquchong(content):
    content = content.decode("utf8")
    # 去除所有标点
    content = re.sub("[\s+\.\!\/_,$%^*():+\"\']+|[+——！，。？?、~@#￥%……&*（）：；【】〔〕“”《》]+".decode("utf8"),
                     "".decode("utf8"), content)
    return content


def preprocess3(content, stop_flag, stopwords):
    """
    :param content: string
    :return: processed content: string 
    """
    content = content.decode("utf8")
    content = re.sub("[0-9]+|[\"]{3}|。$|\"$|。\"$".decode("utf8"), " ".decode("utf8"), content)

    # 高级带过滤分词
    words = jieba.posseg.cut(content)
    result = []
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


def content2vec(corpus, tests):
    """
    :param corpus:
    :param tests:
    :return: tfidf_matrix, new_matrix: two sparse matrix
    """
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    print 'shape of corpus_matrix:\n', tfidf_matrix.shape
    words = tfidf_vectorizer.get_feature_names()  # 获得所有词
    new_matrix = tfidf_vectorizer.transform(tests)
    print 'shape of new_matrix:\n', new_matrix.shape
    return tfidf_matrix, new_matrix


def calres(tfidf_matrix, new_matrix):
    sim_matrix = cosine_similarity(new_matrix, tfidf_matrix)
    return sim_matrix


def resbysim(sim_matrix, old_ids):
    res = []
    test_length = sim_matrix.shape[0]
    for j in range(test_length):
        # print 'j:', j
        sims = sim_matrix[j, :]
        max20 = np.argsort(-sims).tolist()[:50]
        max20 = [old_ids[i] for i in max20]
        # max20 = [i + 1 for i in max20]
        res.append(max20)
    return res


def fun_bm25(corpus, test_corpus):
    dictionary = corpora.Dictionary(corpus)
    content_vectors = [dictionary.doc2bow(content) for content in corpus]
    vector_sorted = sorted(content_vectors[0], key=lambda (x, y): y, reverse=True)

    bm25Model = BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    all_scores = []
    for query in test_corpus:
        scores = bm25Model.get_scores(query, average_idf)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    return all_scores


def findnowid(origin_id, old_ids):
    if origin_id == 367410:
        return 747
    for i, id in enumerate(old_ids):
        # print id,origin_id
        if id == origin_id:
            return i  # 这里return的i没有加1,从0开始数
    print origin_id, "not in!!"
    exit(1)


if __name__ == '__main__':
    stop_flag = ['x']
    stop_words = codecs.open('./data/stopwords.dat', 'r', encoding='utf8').readlines()
    stop_words = [w.strip() for w in stop_words]
    content = "【**绿地控股拟定增募资逾300.1亿元布局多元化产业**】绿地控股发布定增预案，公司拟以不低于14.51元/股非公开发行不超过20.78亿股，募集资金总额不超过301.5亿元，拟分别用于房地产投资、金融投资及偿还银行贷款；发行完成后公司仍不存在控股股东和实际控制人。公司股票将于12月9日复牌。"
    print content
    content = content.decode("utf8")
    content = re.sub("[0-9]+".decode("utf8"), " ".decode("utf8"), content)
    words = jieba.posseg.cut(content)
    result = []
    for word, flag in words:
        if flag not in stop_flag and word not in stop_words:
            result.append(word)
    print ' '.join(result)
