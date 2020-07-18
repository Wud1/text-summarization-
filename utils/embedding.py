import os
import pickle
import time
import re
import numpy as np
from gensim.models import KeyedVectors

from utils.config import *


def timeit(f):
    def wrapper(*args, **kwargs):
        print('[{}] 开始!'.format(f.__name__))
        start_time = time.time()
        res = f(*args, **kwargs)
        end_time = time.time()
        print("[%s] 完成! -> 运行时间为：%.8f" % (f.__name__, end_time - start_time))
        return res

    return wrapper


def load_embedding_matrix(w2v_model_path, vocab_path, vocab_size, embed_size):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return embedding_matrix:
    """
    # word2vec_dict = load_pkl(params['word2vec_output'])
    w2v = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    vocab_dict = open(vocab_path, encoding='utf-8').readlines()
    print('[load_word2vec]:vocab_dict.len:{}'.format(len(vocab_dict)))
    embedding_matrix = np.zeros((vocab_size, embed_size))

    for line in vocab_dict[:vocab_size]:
        word_id = line.split()
        if len(word_id) < 2:
            print('empty word:{}'.format(line))
            continue
        word, i = word_id
        if word in w2v.vocab:
            embedding_matrix[int(i)] = w2v[word]
        else:
            embedding_matrix[int(i)] = np.random.uniform(-10, 10, 256)
    print('embedding_m.shape:{}'.format(embedding_matrix.shape))
    return embedding_matrix