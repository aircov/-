# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:13
@author : 姚明伟
"""
from pytrie import StringTrie

import pypinyin
from flask_script import Manager

from info import create_app

app = create_app('development')  # 开发
# app = create_app('production')  # 上线

manager = Manager(app)


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

    return ''.join([i[0] for i in pinyin(text).split(' ') if len(i) > 0]).replace(' ', '').lower()


# 获取拼音的第一个首字母
def get_all_pinying(text):
    """
        :param text: 文本
        :return: 文本转拼音
        """
    gap = ''
    piny = gap.join(pypinyin.lazy_pinyin(text)).replace(' ', '').lower()

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


if __name__ == '__main__':
    manager.run()
