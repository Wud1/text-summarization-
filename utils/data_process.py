import pandas as pd
import jieba
import re
import numpy as np

from W2V.Word2V import model_wv

# 数据
traing_path = '../DATA/AutoMaster_TrainSet.csv'
test_path = '../DATA/AutoMaster_TestSet.csv'
stop_words = '../DATA/stop_word.txt'

train_df = pd.read_csv(traing_path)
test_df = pd.read_csv(test_path)

train_df.dropna(subset=['Question','Dialogue','Report'],how='any',inplace=True)
test_df.dropna(subset=['Question','Dialogue'],how='any',inplace=True)

# 使用正则表达式去除无用的符号、词语
def clean_symbol(sentence):
    if isinstance(sentence,str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_+|车主说|技师说|语音|图片|你好|您好|nan,，$%^*(+\"\')]+|[:：+——()?【】“”！，,,。？、~@#￥%……&*（）]',
            '', sentence)
    else:
        return ''

# 添加自定义词表
jieba.load_userdict('../DATA/user_dict.txt')


# 去掉停用词
def load_stop_words(stop_word_path):
    file = open(stop_word_path,'r')
    stop_words = file.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

stop_word = load_stop_words(stop_word_path ='../DATA/stop_word.txt')

def fileter_stopwords(words):
    return [word for word in words if word not in stop_word]


def sentence_proc(sentence):

    sentence = clean_symbol(sentence)
    words = jieba.cut(sentence)
    words = fileter_stopwords(words)

    return ' '.join(words)

def data_frame_proc(res):
    for col_name in ['Brand','Model','Question','Dialogue']:
        res[col_name] = res[col_name].apply(sentence_proc)

    if 'Report' in res.columns:
        res['Report'] = res['Report'].apply(sentence_proc)

# return:处理好的数据集
    return res

train_df = data_frame_proc(train_df)
test_df = data_frame_proc(test_df)

train_df.to_csv('../DATA/train_seg_data.csv',index=None , header=True)
test_df.to_csv('../DATA/test_seg_data.csv',index=None,header=True)

train_df['merged'] = train_df[['Question','Dialogue','Report']].apply(lambda x:' '.join(x),axis=1)
test_df['merged'] = test_df[['Question','Dialogue']].apply(lambda x:' '.join(x),axis=1)

merged_df = pd.concat([train_df[['merged']],test_df[['merged']]],axis=0)

# 数据处理完成后保存
merged_df.to_csv('../DATA/train_test_data.csv',index=None,header=True)

train_df['X'] = train_df[['Question','Dialogue']].apply(lambda x: ' '.join(x),axis=1)
test_df['X'] = test_df[['Question','Dialogue']].apply(lambda x: ' '.join(x),axis=1)


#  建立vocab词表
vocab = {word: index for index,word in enumerate(model_wv.wv.index2word)}
reverse_vocab = {index: word for index,word in enumerate(model_wv.wv.index2word)}

def pad_proc(sentence,max_len,vocab):

    # 按空格统计切分词
    words = sentence.strip().split(' ')
    # 截取规定长度的词数
    words = words[:max_len]
    # 填充<unk>,判断是否在vocab中，不在填充<UNK>
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 填充 <start> 和 <end>
    sentence = ['<start>'] + sentence + ['<stop>']
    sentence = sentence + ['<pad>'] * (max_len + 2 - len(words))

    return ' '.join(sentence)

def get_max_len(data):

    max_lens = data.apply(lambda x: x.count(' '))
    return int(np.mean(max_lens) + 2 * np.std(max_lens))

# 获取输入数据 适当的最大长度
train_y_max_len = get_max_len(train_df['X'])
test_y_max_len = get_max_len(test_df['X'])

x_max_len = max(train_y_max_len,test_y_max_len)

train_y_max_len = get_max_len(train_df['Report'])


# 训练集X处理
train_df['X'] = train_df['X'].apply(lambda x:pad_proc(x,x_max_len,vocab))
# 训练集Y处理
train_df['Y'] = train_df['Report'].apply(lambda x:pad_proc(x,train_y_max_len,vocab))
# 测试集X
test_df['X'] = test_df['X'].apply(lambda x:pad_proc(x,x_max_len,vocab))

train_x_pad_path = '../DATA/train_X_pad_data.csv'
train_Y_pad_path = '../DATA/train_Y_pad_data.csv'
test_x_pad_path = '../DATA/test_X_pad_data.csv'

train_df['X'].to_csv(train_x_pad_path,index=None,header=True)
train_df['Y'].to_csv(train_Y_pad_path,index=None,header=True)
test_df['X'].to_csv(test_x_pad_path,index=None,header=True)

unk_index = vocab['<UNK>']
def transform_data(sentence,vocab):
    # 字符串切分成词
    words = sentence.split(' ')
    ids = [vocab[word] if word in vocab else unk_index for word in words]
    return ids

train_ids_x = train_df['X'].apply(lambda x:transform_data(x,vocab))
train_ids_y = train_df['Y'].apply(lambda x:transform_data(x,vocab))
test_ids_x = test_df['X'].apply(lambda x:transform_data(x,vocab))
#
train_data_X = np.array(train_ids_x.tolist())
train_data_Y = np.array(train_ids_y.tolist())
test_data_X = np.array(test_ids_x.tolist())
