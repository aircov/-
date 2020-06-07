# -*- coding: utf-8 -*-
"""
@time   : 2020/06/01 20:51
@author : 姚明伟
https://blog.csdn.net/weixin_42608414/article/details/88046380
使用Python和sklearn来实现一下文本的多分类实战开发
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
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示符号

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

# N = 3
# for cat, cat_id in sorted(cat_to_id.items()):
#     features_chi2 = chi2(features, labels == cat_id)
#     indices = np.argsort(features_chi2[0])
#     feature_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#     print("# '{}':".format(cat))
#     print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
#     print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

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


"""
接下来我们尝试不同的机器学习模型,并评估它们的准确率，我们将使用如下四种模型:
Logistic Regression(逻辑回归)
(Multinomial) Naive Bayes(多项式朴素贝叶斯)
Linear Support Vector Machine(线性支持向量机)
Random Forest(随机森林)
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# 箱体图
import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
"""
从可以箱体图上可以看出随机森林分类器的准确率是最低的，因为随机森林属于集成分类器(有若干个子分类器组合而成)，
一般来说集成分类器不适合处理高维数据(如文本数据),因为文本数据有太多的特征值,使得集成分类器难以应付,另外三个分类器的
平均准确率都在80%以上。其中线性支持向量机的准确率最高。
"""

print(cv_df.groupby('model_name').accuracy.mean())
# model_name
# LinearSVC                 0.855110
# LogisticRegression        0.839418
# MultinomialNB             0.773005
# RandomForestClassifier    0.546248
# Name: accuracy, dtype: float64

# 模型的评估
# 下面我们就针对平均准确率最高的LinearSVC模型，我们将查看混淆矩阵，并显示预测标签和实际标签之间的差异。

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 训练模型
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.33, stratify=labels,
                                                                                 random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 生成混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df.cat.values, yticklabels=cat_id_df.cat.values)
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)
plt.show()

"""
混淆矩阵的主对角线表示预测正确的数量,除主对角线外其余都是预测错误的数量.从上面的混淆矩阵可以看出"蒙牛"类预测最准确,
只有一例预测错误。“平板”和“衣服”预测的错误数量教多。

 多分类模型一般不使用准确率(accuracy)来评估模型的质量,因为accuracy不能反应出每一个分类的准确性,因为当训练数据不平衡
(有的类数据很多,有的类数据很少)时，accuracy不能反映出模型的实际预测精度,这时候我们就需要借助于F1分数、ROC等指标来评估模型。

下面我们将查看各个类的F1分数.
"""
from sklearn.metrics import classification_report

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=cat_id_df['cat'].values))
# accuracy 0.8579358949604171
#               precision    recall  f1-score   support
#
#           书籍       0.95      0.90      0.92      1271
#           平板       0.73      0.78      0.76      3300
#           手机       0.91      0.79      0.84       767
#           水果       0.89      0.82      0.85      3300
#          洗发水       0.78      0.83      0.80      3300
#          热水器       0.86      0.49      0.63       190
#           蒙牛       1.00      0.97      0.98       671
#           衣服       0.83      0.88      0.86      3300
#          计算机       0.94      0.86      0.90      1317
#           酒店       0.98      0.97      0.98      3300
#
#     accuracy                           0.86     20716
#    macro avg       0.89      0.83      0.85     20716
# weighted avg       0.86      0.86      0.86     20716

"""
从以上F1分数上看,"蒙牛"类的F1分数最大(只有一个预测错误)，“热水器”类F1分数最差只有66%，
究其原因可能是因为“热水器”分类的训练数据最少只有574条,使得模型学习的不够充分,导致预测失误较多吧。

下面我们来查看一些预测失误的例子,希望大家能从中发现一些奥秘,来改善我们的分类器。
"""

from IPython.display import display

for predicted in cat_id_df.cat_id:
    for actual in cat_id_df.cat_id:
        if predicted != actual and conf_mat[actual, predicted] >= 6:
            print("{} 预测为 {} : {} 例.".format(id_to_cat[actual], id_to_cat[predicted], conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['cat', 'review']])
            print('')
