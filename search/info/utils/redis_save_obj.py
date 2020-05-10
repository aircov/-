# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 11:42
@author : 姚明伟
"""

import pickle

from config import conn_redis as r

class Duankeke(object):
    def keke(self):
        print(33333)
        return "haha"


a = Duankeke()

# pickle模块将class转化为str，再反序列化回来
r.set('duan',pickle.dumps(a))
result = r.get('duan')
result = pickle.loads(result)
print(result)
# 这里打印的是object class


a = result.keke()  # 正常打印33333
print(a)  # haha