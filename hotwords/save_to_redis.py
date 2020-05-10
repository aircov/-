# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 17:25
@author : 姚明伟
# mysql数据更新到redis
"""
import datetime

import pymysql
import redis

# mysql链接
db = pymysql.connect(host="127.0.0.1", port=3306, user='root', password='mysql', db='information')
cursor = db.cursor()

# redis链接
r = redis.StrictRedis('127.0.0.1', 6379, 0, decode_responses=True)


def main():
    sql = """select query,heat from hot_words"""
    cursor.execute(sql)
    ret = dict(cursor.fetchall())  # {'香港影院复工': '222601', '海军航空兵飞行餐太诱人': '222401'}

    # 管道
    pl = r.pipeline()
    for i in ret:
        pl.hset("hot_word_heat", i, int(ret[i]))
    # 删除之前保存的
    r.delete("hot_word_heat")
    pl.execute()
    print("数据保存到redis成功{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # print(r.hgetall("hot_word_heat"))
    print(len(r.hgetall("hot_word_heat")))


if __name__ == '__main__':
    main()
