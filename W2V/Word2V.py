from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from utils import config

#  读取数据
merger_data_path ='../data/train_test_data.csv'

# 使用word2ver训练词向量
model_wv = word2vec.Word2Vec(LineSentence(merger_data_path)
                             ,workers=8,
                             min_count=5,
                             size=200,
                             iter=config.EPOCH_NUM,)



embedding_matrix = model_wv.wv.vectors

train_x_pad_path = '../data/train_X_pad_data.csv'
train_Y_pad_path = '../data/train_Y_pad_data.csv'
test_x_pad_path = '../data/test_X_pad_data.csv'

model_wv.build_vocab(LineSentence(train_x_pad_path),update=True)
model_wv.train(LineSentence(train_x_pad_path),epochs=5,total_examples=model_wv.corpus_count)

model_wv.build_vocab(LineSentence(train_Y_pad_path),update=True)
model_wv.train(LineSentence(train_Y_pad_path),epochs=5,total_examples=model_wv.corpus_count)

model_wv.build_vocab(LineSentence(test_x_pad_path),update=True)
model_wv.train(LineSentence(test_x_pad_path),epochs=5,total_examples=model_wv.corpus_count)

model_w2v = '../data/model_w2v.model'

# 保存词向量模型
model_wv.save(model_w2v)

# 更新vocab
vocab = {word : index for index, word in enumerate(model_wv.wv.index2word)}
reverse_vocab = {index : word for index, word in enumerate(model_wv.wv.index2word)}

embedding_matrix = model_wv.wv.vectors
