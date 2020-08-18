# -*- coding: utf-8 -*-
'''
获取百度热搜词
# 每小时执行一次热搜词爬虫脚本
0 */1 * * * /usr/bin/python3 /root/recmd_proj/hotwords/baidu_weibo_hot_words.py >> /root/recmd_proj/hotwords/log.log 2>&1
'''
import datetime
import time

from elasticsearch import Elasticsearch, helpers
import requests
from bs4 import BeautifulSoup
import re
import pymysql

db = pymysql.connect(host="127.0.0.1", port=3306, user='root', password='mysql', db='information')
cursor = db.cursor()
es = Elasticsearch("127.0.0.1")


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
					   "crawl_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "update_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"source": "百度"}

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
				heat = ["99999999"]
			resDict = {"num": str(i + 1), "query": query[0], "heat": heat[0], "url": jumpHref,
					   "crawl_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "update_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"source": "微博"}

			resDictList.append(resDict)
		i = i + 1
	return (resDictList)


def save_to_mysql(data):
	for i in data:
		keys = ", ".join(i.keys())
		values = ", ".join(['%s'] * len(i))

		# 去重
		sel_sql = """select * from hot_words where query='{}' and source = '{}'""".format(i.get("query"),i.get("source"))
		cursor.execute(sel_sql)
		sel_data = cursor.fetchall()
		if len(sel_data)>0:
			# 有重复数据
			columns = [column[0] for column in cursor.description]
			datas = []
			for row in sel_data:
				datas.append(dict(zip(columns, row)))
			print(datas)
			# 如果数据库中的heat值小于本次爬取的，更新数据
			if int(datas[0].get('heat'))<int(i.get('heat')):
				up_sql = """update hot_words set num='{}', heat='{}',update_time='{}'  where query='{}' and source = '{}'""".format(i.get("num"),i.get("heat"),i.get("update_time"),i.get("query"),i.get("source"))
				cursor.execute(up_sql)
				db.commit()
			else:
				pass

		else:
			# 没有重复数据，插入
			insert_sql = """insert into {} ({}) value ({})""".format("hot_words", keys, values)
			try:
				ret = cursor.execute(insert_sql, tuple(i.values()))
				db.commit()
				# print("Successful")
			except Exception as e:
				print("Failed", e, insert_sql, tuple(i.values()))
				db.rollback()

	print("%s 数据保存成功!" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def built_es_index():
	# 建立es索引，映射  设置分片、
	mapping = {
		'properties': {
			'query': {
				'type': 'text',
				'analyzer': 'ik_max_word',
				'search_analyzer': 'ik_max_word',
				"fields": {
					"keyword": {             #name.keyword keyword类型，用于排序和聚合
						"type": "keyword"
					}
				}
			},
			'heat':{
				'type':'integer'
			},
			'crawl_time': {
				'type': 'date',
				"format":'yyyy-MM-dd HH:mm:ss'   # 设置日期格式
			},
			'update_time': {
				'type': 'date',
				"format":'yyyy-MM-dd HH:mm:ss'   # 设置日期格式
			},
		}
	}
	es.indices.delete(index='hot_words', ignore=[400, 404])
	es.indices.create(index='hot_words', ignore=400,
					  body={
							"index":{
								"number_of_shards": 5,  # 分片个数，在创建索引不指定时 默认为5， 可根据集群node数量合理设置分片个数，从而提高数据查询、读取效率
								"number_of_replicas": 1  # 数据副本，一般设置为 1
							}
						}
					  )
	result = es.indices.put_mapping(index='hot_words', doc_type='doc', body=mapping)

	# 设置max_result_window
	es.indices.put_settings(
		index="hot_words",
		body={
			"index":{
				"max_result_window":500000,
			}
		}
	)

	print(result)


def read_mysql_to_es():
	# 保存数据到es,全量更新
	start_time = time.time()
	sql = """select num,query,heat,url,crawl_time,source from hot_words"""
	cursor.execute(sql)
	columns = [column[0] for column in cursor.description]
	datas = []
	for row in cursor.fetchall():
		datas.append(dict(zip(columns, row)))

	# print(datas)
	actions = []
	for data in datas:
		action = {
			'_index': 'hot_words',
			'_type': 'doc',
			'_source': {
				'num': data.get('num'),
				'query': data.get('query'),
				'heat': data.get('heat'),
				'url': data.get('url'),
				'crawl_time': data.get('crawl_time'),
				'update_time': data.get('update_time'),
				'source': data.get('source'),
			}
		}
		actions.append(action)

	helpers.bulk(es, actions=actions, raise_on_error=True)
	end_time = time.time()
	print('导入数据耗时：', (end_time - start_time))


def generator():
	# 保存数据到es，增量更新
	start_time = time.time()
	sql = """select id,num,query,heat,url,crawl_time,update_time,source from hot_words"""
	cursor.execute(sql)
	columns = [column[0] for column in cursor.description]
	datas = []
	for row in cursor.fetchall():
		datas.append(dict(zip(columns, row)))
	print(len(datas))
	for data in datas:
		yield {
			'_op_type': 'create',
			'_id': data.get('id'),
			'_source': {
				'num': data.get('num'),
				'query': data.get('query'),
				'heat': data.get('heat'),
				'url': data.get('url'),
				'crawl_time': data.get('crawl_time'),
				'update_time': data.get('update_time'),
				'source': data.get('source'),
			}
		}
	end_time = time.time()
	print('导入数据耗时：', (end_time - start_time))

if __name__ == '__main__':
	res = getbaiduHotWord()
	save_to_mysql(res)

	ret = getweiboHotWord()
	save_to_mysql(ret)

	# 第一次运行建立es——index
	built_es_index()

	# 保存到es
	# read_mysql_to_es()
	helpers.bulk(es,generator(),index='hot_words',doc_type='doc',raise_on_exception=False, raise_on_error=False)
