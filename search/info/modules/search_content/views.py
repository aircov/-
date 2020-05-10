# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:14
@author : 姚明伟
"""

import json

from flask import request, jsonify, make_response

from info.utils.content import search_content_indistinct, search_content_exact
from info.utils.thread_return_value import MyThread
from loggers import *
from config import conn_redis as r

from tools.response_code import RET
from . import search_content_blu


def convert(data):
    if isinstance(data, bytes):
        return data.decode('ascii')
    if isinstance(data, dict):
        return dict(map(convert, data.items()))
    if isinstance(data, tuple):
        return map(convert, data)
    return data


@search_content_blu.route('/', methods=['GET'])
def search_person():
    # 获取参数
    start_time = time.time()
    keyword = request.args.get('keyword')
    page = request.args.get('page')
    limit = request.args.get('limit')
    # print(request.remote_addr)
    print(keyword)
    if not all([keyword, page, limit]):
        warn(json.dumps({'code': RET.PARAMERR, 'errmsg': '参数错误'}, ensure_ascii=False))
        return jsonify(errno=RET.PARAMERR, errmsg='参数错误')
    # 校验参数limit类型
    try:
        limit = int(limit)
    except Exception as e:
        warn(json.dumps({'code': RET.PARAMERR, 'errmsg': '参数错误'}, ensure_ascii=False))
        return jsonify(errno=RET.PARAMERR, errmsg=e)

    # 模糊匹配结果
    st = MyThread(target=search_content_indistinct, args=(keyword, page, limit))
    # 精确匹配结果
    st2 = MyThread(target=search_content_exact, args=(keyword, page, limit))
    st.start()
    st2.start()

    # 获取线程返回值
    indistinct_people = st.get_result()
    exact_people = st2.get_result()

    ret = dict()
    ret['indistinct_people'] = indistinct_people
    ret['exact_people'] = exact_people.get('hits').get('hits') if exact_people.get('hits').get('hits') else []

    end_time = time.time()

    info(json.dumps({"code":RET.OK, "message":'OK', "body":ret,"keyword":keyword,"page":page,"limit":limit,"total_time":end_time-start_time},ensure_ascii=False))
    return jsonify(code=RET.OK, message='OK', body=ret)


# 构造字典树
from info.modules.search_content import search_script_conf


@search_content_blu.route('/su/')
def index():
    """
    搜索提示功能
    根据输入的值自动联想,支持中文,英文,英文首字母
    :return: response
    """

    # reload(search_script_conf)
    # print(search_script_conf.sug)
    start_time = time.time()
    wd = request.args.get('wd')

    if not wd:
        return make_response("""queryList({"q":"","p":false,"bs":"","csor":"0","status":770,"s":[]});""")

    # 搜索词(支持中文，英文，英文首字母)
    s = wd

    # 返回15个
    result = search_script_conf.get_tips_word(search_script_conf.sug, search_script_conf.data, s)
    print('前缀：',result)

    if len(result)>0:
        # 从redis获取热度值
        heat_list = r.hmget("hot_word_heat",result)
        _map = dict(zip(result,heat_list))
        # 按照热度值排序
        data = dict(sorted(_map.items(), key=lambda x: int(x[1]), reverse=True))
        print("热度值排序:",data)
        result = list(data.keys())[:15]

    response = make_response(
        """queryList({'q':'""" + wd + """','p':false,'s':""" + str(result) + """});""")

    response.headers['Content-Type'] = 'text/javascript; charset=utf-8'

    end_time = time.time()
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
    ret['total_time'] = end_time-start_time
    info(json.dumps(ret, ensure_ascii=False))
    return response
