# -*- coding: utf-8 -*-
"""
@time   : 2020/06/01 20:51
@author : 姚明伟
https://blog.csdn.net/weixin_42608414/article/details/88046380
"""

import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jieba as jb
import pkuseg
import re

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# plt参数设置
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
# plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示符号

# 加载用户自定义词典
user_dict = ["海飞丝"]
seg = pkuseg.pkuseg(user_dict=user_dict)

df = pd.read_csv('./data/online_shopping_10_cats.csv')
df = df[['cat', 'review']]
print("数据总量: %d ." % len(df))

print(df.sample(10))

# 数据清洗
print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
df[df.isnull().values==True]
df = df[pd.notnull(df['review'])]

# 统计各个类别的数据量
d = {'cat':df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

# 用图形化的方式再查看一下各个类别的分布
df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
plt.title("类目数量分布")
plt.ylabel('数量', fontsize=18)
plt.xlabel('类目', fontsize=18)
# plt.show()

# 数据预处理
# 接下来我们要将cat类转换成id，这样便于以后的分类模型的训练
df['cat_id'] = df['cat'].factorize()[0]
cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)
print(df.sample(10))

print(cat_id_df)


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")  # 正则
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 加载停用词
stopwords = stopwordslist("./data/chineseStopWords.txt")

#删除除字母,数字，汉字以外的所有符号
df['clean_review'] = df['review'].apply(remove_punctuation)
print(df.sample(10))
# 我们过滤掉了review中的标点符号和一些特殊符号，并生成了一个新的字段 clean_review。
# 接下来我们要在clean_review的基础上进行分词,把每个评论内容分成由空格隔开的一个一个单独的词语。

#分词，并过滤停用词
# df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in seg.cut(x) if w not in stopwords]))
print(df.head())

# 经过分词以后我们生成了cut_review字段。在cut_review中每个词语中间都是由空格隔开，接下来我要在cut_review的基础上生成每个分类的词云，
# 我们要在每个分类中罗列前100个高频词。然后我们要画出这些高频词的词云

"""
接下来我要计算cut_review的 TF-IDF的特征值，TF-IDF（term frequency–inverse document frequency）是一种用于
信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。
TF-IDF是在单词计数的基础上，降低了常用高频词的权重,增加罕见词的权重。因为罕见词更能表达文章的主题思想,比如在一篇文章中出现
了“中国”和“卷积神经网络”两个词,那么后者将更能体现文章的主题思想,而前者是常见的高频词,它不能表达文章的主题思想。所以“卷积
神经网络”的TF-IDF值要高于“中国”的TF-IDF值。这里我们会使用sklearn.feature_extraction.text.TfidfVectorizer方法来
抽取文本的TF-IDF的特征值。这里我们使用了参数ngram_range=(1,2),这表示我们除了抽取评论中的每个词语外,还要抽取每个词相邻
的词并组成一个“词语对”,如: 词1，词2，词3，词4，(词1，词2)，(词2,词3)，(词3，词4)。这样就扩展了我们特征集的数量,有了丰富
的特征集才有可能提高我们分类文本的准确度。参数norm='l2',是一种数据标准划处理的方式,可以将数据限制在一点的范围内比如说(-1,1)
"""


tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(df.cut_review)
labels = df.cat_id
print(features.shape)
print('-----------------------------')
print(features)

"""
我们看到我们的features的维度是(62773,657425),这里的62773表示我们总共有62773条评价数据，657425表示我们的特征数量
这包括全部评论中的所有词语数+词语对(相邻两个单词的组合)的总数。下面我们要是卡方检验的方法来找出每个分类中关联度最大的
两个词语和两个词语对。卡方检验是一种统计学的工具,用来检验数据的拟合度和关联度。在这里我们使用sklearn中的chi2方法。
"""

N = 3
for cat, cat_id in sorted(cat_to_id.items()):
    features_chi2 = chi2(features, labels == cat_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(cat))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

"""
我们可以看到经过卡方(chi2)检验后，找出了每个分类中关联度最强的两个词和两个词语对。这些词和词语对能很好的反映出分类的主题
"""


# 朴素贝叶斯分类器
X_train, X_test, y_train, y_test = train_test_split(df['cut_review'], df['cat_id'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# 保存模型
joblib.dump(clf, "naive_bayes_train_model.m")
# 加载模型
# clf = joblib.load("naive_bayes_train_model.m")


def myPredict(sec):
    # 预测函数
    new_sec = remove_punctuation(sec)
    print(new_sec)
    format_sec=" ".join([w for w in seg.cut(new_sec) if w not in stopwords])
    pred_cat_id=clf.predict(count_vect.transform([format_sec]))
    print(id_to_cat[pred_cat_id[0]])

myPredict("感谢京东自营产地直采。你们把握质量关。第三次购买")
myPredict("头屑越洗越多，下次再也不买了。")







