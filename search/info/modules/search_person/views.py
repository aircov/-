import json
import re
import pandas as pd
from importlib import reload

from flask import request, jsonify, make_response

# from info.utils.content import search_person_indistinct, search_person_exact
from info.utils.thread_return_value import MyThread
from loggers import *

from tools.response_code import RET
from . import search_person_blu


def convert(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    if isinstance(data, dict):
        return dict(map(convert, data.items()))
    if isinstance(data, tuple):
        return map(convert, data)
    return data


# 关键字搜索人  GET  http://192.168.191.1:8000/api/search/person/?keyword=丸子&user_id=10004002692910844800290980133243&app_id=kk&limit=3
# @search_person_blu.route('/', methods=['GET'])
# def search_person():
#     """
#     搜人：
#     1.获取参数
#     2.校验参数
#     3.判断参数类型
#     4.精确匹配
#     5.模糊匹配，人气值、分数排序
#     6.返回数据
#     :return: 精确匹配结果+模糊匹配结果(按照人气值倒序排序)
#     """
#     # 获取参数
#     # start_time = time.time()
#     keyword = request.args.get('keyword')
#     page = request.args.get('page')
#     limit = request.args.get('limit')
#     # print(request.remote_addr)
#     if not all([keyword, page, limit]):
#         warn(json.dumps({'code': RET.PARAMERR, 'errmsg': '参数错误'}, ensure_ascii=False))
#         return jsonify(errno=RET.PARAMERR, errmsg='参数错误')
#     # 校验参数limit类型
#     try:
#         limit = int(limit)
#     except Exception as e:
#         warn(json.dumps({'code': RET.PARAMERR, 'errmsg': '参数错误'}, ensure_ascii=False))
#         return jsonify(errno=RET.PARAMERR, errmsg=e)
#
#
#     # 模糊匹配结果
#     st = MyThread(target=search_person_indistinct, args=(keyword,))
#     # 精确匹配结果
#     st2 = MyThread(target=search_person_exact, args=(keyword,))
#     st.start()
#     st2.start()
#
#     # 获取线程返回值
#     indistinct_people = st.get_result()
#     exact_people = st2.get_result()
#
#     ret = dict()
#     ret['indistinct_people'] = indistinct_people['hits']['hits']
#     ret['exact_people'] = exact_people['hits']['hits']
#
#
#     return jsonify(errno=RET.OK, errmsg='OK', body=ret)


# 构造字典树
from info.modules.search_person import search_script_conf

@search_person_blu.route('/su/')
def index():
    """
    搜索提示功能
    根据输入的值自动联想,支持中文,英文,英文首字母
    :return: response
    """

    # reload(search_script_conf)
    # print(search_script_conf.sug)

    wd = request.args.get('wd')

    if not wd:
        return make_response("""queryList({"q":"","p":false,"bs":"","csor":"0","status":770,"s":[]});""")

    # 搜索词(支持中文，英文，英文首字母)
    s = wd

    result = search_script_conf.get_tips_word(search_script_conf.sug, search_script_conf.data, s)[:100]

    response = make_response(
        """queryList({'q':'""" + wd + """','p':false,'s':""" + str(result) + """});""")

    response.headers['Content-Type'] = 'text/javascript; charset=utf-8'

    # 记录日志
    ret = dict()
    ret['code'] = 200
    ret['msg'] = "ok"
    ret['search_word'] = wd
    ret['search_result'] = result
    ret['search_type'] = 'search_tips'
    ret['gmt_created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ret['user_id'] = ''
    ret['platformCode'] = ''
    info(json.dumps(ret, ensure_ascii=False))
    return response
