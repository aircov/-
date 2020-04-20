# -*- coding: utf-8 -*-
'''
获取百度热搜词
Author:yuzg667
https://github.com/yuzg667/hotwords
'''
import datetime

import requests
from bs4 import BeautifulSoup
import re
import pymysql


def getbaiduHotWord():
    header = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Host": "top.baidu.com",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"

    }
    url = "http://top.baidu.com/buzz?b=1&fr=topindex"
    r = requests.get(url, headers=header)
    html = r.text.encode('iso-8859-1').decode('gbk')  # 中文乱码处理，非常重要    print (html)

    s = BeautifulSoup(html, "lxml")
    # 获取页面内所有的热搜词的html
    hotwords = s.find_all("tr")  # oneline

    i = 0
    resDictList = []
    for hotword in hotwords:
        # 正则取
        hotword = str(hotword)  # 正则必须是string
        patternQuery = re.compile(r'''target="_blank">(.*?)</a>''')
        query = re.findall(patternQuery, hotword)

        patternHeat = re.compile(r'''">(.*?)</span>''')
        heat = re.findall(patternHeat, hotword)

        if len(query) > 0 and len(heat) > 0:  # 有返回空的情况，过滤掉

            jumpHref = "https://www.baidu.com/baidu?cl=3&tn=SE_baiduhomet8_jmjb7mjw&rsv_dl=fyb_top&fr=top1000&wd=" + str(
                query[0])
            resDict = {"num": str(i + 1), "query": query[0], "heat": heat[-1], "url": jumpHref,
                       "crawl_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            resDictList.append(resDict)
            i = i + 1
    return (resDictList)


def getweiboHotWord():
    header = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "Host": "s.weibo.com",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36"

    }
    url = "https://s.weibo.com/top/summary?cate=realtimehot"  # 请求地址
    r = requests.get(url, headers=header)
    html = r.text

    s = BeautifulSoup(html, "lxml")
    # 获取页面内所有的热搜词的html
    hotwords = s.find_all("td", class_="td-02")

    i = 0
    resDictList = []
    for hotword in hotwords:
        # 正则取
        hotword = str(hotword)  # 正则必须是string

        patternQuery = re.compile(r'''target="_blank">(.*?)</a>''')
        query = re.findall(patternQuery, hotword)

        patternHeat = re.compile(r'''<span>(.*?)</span>''')
        heat = re.findall(patternHeat, hotword)

        patternJumpHref = re.compile(r'''<a href="(.*?)" target="_blank">''')
        jumpHref = re.findall(patternJumpHref, hotword)

        if len(query) > 0 and len(jumpHref) > 0:  # 有返回空的情况，过滤掉

            jumpHref = "https://s.weibo.com" + str(jumpHref[0])
            if len(heat) < 1:
                heat = ["置顶"]
            resDict = {"num": str(i + 1), "query": query[0], "heat": heat[0], "url": jumpHref,
                       "crawl_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

            resDictList.append(resDict)
        i = i + 1
    return (resDictList)


def save_to_mysql(data):
    db = pymysql.connect(host="127.0.0.1", port=3306, user='root', password='mysql', db='information')
    cursor = db.cursor()

    for i in data:
        keys = ", ".join(i.keys())
        values = ", ".join(['%s'] * len(i))

        # 删除
        del_sql = """delete from hot_words where query='{}'""".format(i.get("query"))
        cursor.execute(del_sql)
        db.commit()

        # 插入
        insert_sql = """insert into {} ({}) value ({})""".format("hot_words", keys, values)
        try:
            ret = cursor.execute(insert_sql, tuple(i.values()))
            db.commit()
            print("Successful")
        except Exception as e:
            print("Failed", e, insert_sql, tuple(i.values()))
            db.rollback()

    print("数据保存成功")


def read_mysql_to_es():
    # 保存数据到mysql
    pass


if __name__ == '__main__':
    res = getweiboHotWord()
    # print(res)
    save_to_mysql(res)

    ret = getbaiduHotWord()
    # print(ret)
    save_to_mysql(ret)
