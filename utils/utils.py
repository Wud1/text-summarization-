import numpy as np
import tensorflow as tf
from gensim.models import  word2vec

import utils.config
def load_data():
    """
    加载处理好的数据
    :return: 数据
    """
    train_X = np.load(config.train_x_path + '.npy')
    train_Y = np.load(config.train_y_path + '.npy')
    test_X = np.load(config.test_x_path + '.npy')
    return train_X, train_Y, test_X

def config_gpu():
    """
    配置GPU环境
    :return:
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)