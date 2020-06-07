# -*- coding: utf-8 -*-
"""
@time   : 2020/06/07 18:57
@author : 姚明伟
文本分类综合（rnn，cnn，word2vec，TfidfVectorizer）
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
#KeyedVectors实现实体（单词、文档、图片都可以）和向量之间的映射，实体都用string id表示
#有时候运行代码时会有很多warning输出，如提醒新版本之类的，如果不想乱糟糟的输出可以这样


# 预训练词向量
# 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('data/sgns.zhihu.bigram',
                                          binary=False)
# 词向量模型
# 在这个词向量模型里，每一个词是一个索引，对应的是一个长度为300的向量，我们今天需要构建的LSTM神经网络模型并不能直接处理汉字文本，
# 需要先进行分次并把词汇转换为词向量，步骤请参考：
# 0.原始文本：我喜欢文学
# 1.分词：我，喜欢，文学
# 2.Tokenize(索引化)：[2,345，4564]
# 3.Embedding(词向量化)：用一个300维的词向量，上面的tokens成为一个[3，300]的矩阵
# 4.RNN:1DCONV,GRU,LSTM等
# 5.经过激活函数输出分类：如sigmoid输出在0到1间
# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = cn_model['山东大学'].shape[0]  #一词山东大学，shape[0]返回行数
print('词向量的长度为{}'.format(embedding_dim))
print(cn_model['山东大学'])

# 计算相似度
print(cn_model.similarity('橘子', '橙子'))

# dot（'橘子'/|'橘子'|， '橙子'/|'橙子'| ），余弦相似度
sim = np.dot(cn_model['橘子']/np.linalg.norm(cn_model['橘子']),
        cn_model['橙子']/np.linalg.norm(cn_model['橙子']))
print("sim: ", sim)

# 找出最相近的词，余弦相似度
print(cn_model.most_similar(positive=['大学'], topn=10))

# 找出不同的词
test_words = '老师 会计师 程序员 律师 医生 老人'
test_words_result = cn_model.doesnt_match(test_words.split())
print('在 '+test_words+' 中:\n不是同一类别的词为: %s' %test_words_result)
# 老人

print(cn_model.most_similar(positive=['女人', '出轨'], negative=['男人'], topn=1))
# [('劈腿', 0.5849197506904602)]

# 训练语料 （数据集）
# 本教程使用了酒店评论语料，训练样本分别被放置在两个文件夹里： 分别的pos和neg，
# 每个文件夹里有2000个txt文件，每个文件内有一段评语，共有4000个训练样本，这样大小的样本数据在NLP中属于非常迷你的

# 获得样本的索引，样本存放于两个文件夹中，
# 分别为 正面评价'pos'文件夹 和 负面评价'neg'文件夹
# 每个文件夹中有2000个txt文件，每个文件中是一例评价，一个对一个
import os
pos_txts = os.listdir('data/pos')
neg_txts = os.listdir('data/neg')
print( '样本总共: '+ str(len(pos_txts) + len(neg_txts)) )

# 现在我们将所有的评价内容放置到一个list里
train_texts_orig = [] # 存储所有评价，每例评价为一条string，原始评论
# 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
# 其中前2000条文本为正面评价，后2000条为负面评价
#以下为读入.txt文件过程
for i in range(len(pos_txts)):
    with open('data/pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('data/neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()

# # 我们使用tensorflow的keras接口来建模
# from keras.models import Sequential
# from keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional#Dense全连接
# #Bidirectional双向LSTM  callbacks用来调参
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.optimizers import RMSprop
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)


# 索引长度标准化
# 因为每段评语的长度是不一样的，我们如果单纯取最长的一个评语，并把其他评填充成同样的长度，
# 这样十分浪费计算资源，所以我们取一个折衷的长度。

# 获得所有tokens的长度
num_tokens = [len(tokens) for tokens in train_tokens ]
num_tokens = np.array(num_tokens)
# 平均tokens的长度
np.mean(num_tokens)
# 最长的评价tokens的长度
np.max(num_tokens)

plt.hist(np.log(num_tokens), bins = 100)#有大有小取对数
plt.xlim((0,20))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)

# 取tokens的长度为236时，大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
print(np.sum(num_tokens < max_tokens) / len(num_tokens))

# 反向tokenize
# 为了之后来验证 我们定义一个function，用来把索引转换成可阅读的文本，这对于debug很重要。
# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text

reverse = reverse_tokens(train_tokens[0])
# 经过tokenize再恢复成文本
# 可见标点符号都没有了
print(reverse)
# '早餐太差无论去多少人那边也不加食品的酒店应该重视一下这个问题了房间本身很好'

# 原始文本
print(train_texts_orig[0])
# ‘早餐太差，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。\n\n房间本身很好。’

embedding_dim



