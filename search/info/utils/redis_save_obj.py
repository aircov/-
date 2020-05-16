# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 11:42
@author : 姚明伟
# 保存对象到redis
"""
import json
import pickle

import redis

r = redis.StrictRedis('127.0.0.1',6379,0)
r_decode = redis.StrictRedis('127.0.0.1',6379,0,decode_responses=True)

class Duankeke(object):
    def keke(self):
        print(33333)
        s='hh'
        b='ss'
        return s,b


a = Duankeke()

# pickle模块将class转化为str，再反序列化回来
r.set('duan',pickle.dumps(a))
result = r.get('duan')
result = pickle.loads(result)
print(result)
# 这里打印的是object class


a = result.keke()  # 正常打印33333
print(a)  # haha


from pytrie import StringTrie
import pypinyin
import pandas as pd



#显示所有列
# pd.set_option('display.max_columns', None)



# 文本转拼音
def pinyin(text):
    """
    :param text: 文本
    :return: 文本转拼音
    """
    gap = ' '
    piny = gap.join(pypinyin.lazy_pinyin(text))
    return piny


# 获取拼音的每个首字母
def get_every_word_first(text):
    """
    :param text:文本
    :return: 返回拼音首字母
    """

    return ''.join([i[0] for i in pinyin(text).split(' ') if len(i) > 0]).replace(' ','').lower()


# 获取拼音的第一个首字母
def get_all_pinying(text):
    """
        :param text: 文本
        :return: 文本转拼音
        """
    gap = ''
    piny = gap.join(pypinyin.lazy_pinyin(text)).replace(' ','').lower()

    return piny


# 自定义字典树类
class Suggester(object):
    def __init__(self):
        self.trie = None
        self.trie = StringTrie()

    def update_trie(self, word_list):
        for word in word_list:
            # word = word.lower()
            # 拼音提取，首字母，全拼都改成小写，去空格
            word_pinyin1 = get_every_word_first(word)
            word_pinyin2 = get_all_pinying(word)

            # 拼音建立字典树
            self.trie[word] = word
            self.trie[word_pinyin1] = word_pinyin1
            self.trie[word_pinyin2] = word_pinyin2

    def search_prefix(self, prefix):
        return self.trie.values(prefix=prefix)


# 构建字典树
def build_all_trie(wordlist):
    """
    :param wordlist: 关键词列表
    :return: 字典树和映射数据集
    """
    sug = Suggester()
    sug.update_trie(wordlist)
    data = pd.DataFrame({"word": wordlist})
    data['pinyin1'] = data['word'].apply(lambda x: get_every_word_first(x))
    data['pinyin2'] = data['word'].apply(lambda x: get_all_pinying(x))

    return sug, data


# 判断字符串只包含中文
def check_contain_chinese(check_str):
    flag = True
    for ch in check_str:
        if u'\u4e00' >= ch or ch >= u'\u9fff':
            flag = False
    return flag


# 关键词搜索提示查询
def get_tips_word(sug, data, s):
    """
    :param sug: 字典树
    :param data: 中文和英文映射数据集
    :param s: 搜索词
    :return: 返回搜索提示词
    """
    try:
        if len(s) > 0:
            # 判断输入是否只包含中文，若只中文，按中文查
            if check_contain_chinese(s) is True:
                # 输出结果
                kk = sug.search_prefix(s)
                result3 = data[data['word'].isin(kk)]
                result6 = list(set(result3['word']))
                return result6

            # 若不是只包含中文，转换为英文去查询
            else:
                s1 = get_all_pinying(s)
                kk = sug.search_prefix(s1)
                result1 = data[data['pinyin1'].isin(kk)]
                result2 = data[data['pinyin2'].isin(kk)]
                result3 = data[data['word'].isin(kk)]
                result4 = result1.append(result2, ignore_index=True)
                result5 = result3.append(result4, ignore_index=True)
                # 输出结果
                result6 = list(set(result5['word']))
                return result6
        else:
            return

    except Exception as e:
        print(e)
        pass



final_results = r_decode.hgetall("hot_word_heat")
# print(final_results)
word_list = list(final_results.keys())
# print(word_list)
# 构建字典树
sug, data = build_all_trie(word_list)
# print(data)



# pickle模块将class转化为str，再反序列化回来
r.set('tree',pickle.dumps((sug,data)))



temp = r.get('tree')
# print(temp)
result = pickle.loads(temp)
print(result[1])


ret_tree = get_tips_word(result[0],result[1],'the')
print(ret_tree)