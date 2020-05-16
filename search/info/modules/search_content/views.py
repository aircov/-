# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:14
@author : 姚明伟
"""

import json
import pickle

from flask import request, jsonify, make_response

from info.modules.search_content.search_script_conf import get_tips_word
from info.utils.content import search_content_indistinct, search_content_exact
from info.utils.thread_return_value import MyThread
from loggers import *
from config import r_decode, r

from tools.response_code import RET
from . import search_content_blu

TREE = None


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

    info(json.dumps({"code": RET.OK, "message": 'OK', "body": ret, "keyword": keyword, "page": page, "limit": limit,
                     "total_time": end_time - start_time}, ensure_ascii=False))
    return jsonify(code=RET.OK, message='OK', body=ret)


@search_content_blu.route('/update/', methods=['GET'])
def update_suggest():
    # 从redis中加载一次字典树，更新全局变量
    global TREE

    temp = r.get('tree')
    TREE = pickle.loads(temp)
    result = get_tips_word(TREE[0], TREE[1], 'ig')
    print(result)
    print(TREE)
    return "success"


@search_content_blu.route('/su/')
def index():
    """
    搜索提示功能
    根据输入的值自动联想,支持中文,英文,英文首字母
    :return: response
    """
    start_time = time.time()
    # 输入词转小写
    wd = request.args.get('wd').lower()

    user_id = request.args.get('user_id')
    if user_id and user_id != 'None':
        print(user_id)
        print(type(user_id))

    if not wd:
        return make_response("""queryList({"q":"","p":false,"bs":"","csor":"0","status":770,"s":[]});""")

    # 搜索词(支持中文，英文，英文首字母)
    s = wd

    # result = search_script_conf.get_tips_word(search_script_conf.sug, search_script_conf.data, s)
    #     # print('前缀：',result)
    global TREE
    if TREE is None:
        # 第一次为空，需要在接口中加载一次已经生成好的字典树，pickle.loads这一步耗时接近1s
        temp = r.get('tree')
        TREE = pickle.loads(temp)

    # 内容中有字典树，直接获取
    suggest = get_tips_word(TREE[0], TREE[1], s)
    print('前缀：', suggest)

    data_top = {}
    if len(suggest) > 0:
        # 从redis获取热度值
        heat_list = r_decode.hmget("hot_word_heat", suggest)
        _map = dict(zip(suggest, heat_list))
        # 按照热度值排序
        data = dict(sorted(_map.items(), key=lambda x: int(x[1]), reverse=True))
        print("热度值排序:", data)
        # TODO 获取个性化搜索结果展示
        suggest = list(data.keys())[:15]
        data_top = {i: data[i] for i in suggest}

    response = make_response(
        """queryList({'q':'""" + wd + """','p':false,'s':""" + str(suggest) + """});""")

    response.headers['Content-Type'] = 'text/javascript; charset=utf-8'

    end_time = time.time()
    # 记录日志
    ret = dict()
    ret['code'] = 200
    ret['msg'] = "ok"
    ret['search_word'] = wd
    ret['search_suggest'] = suggest
    ret['heat_rank'] = data_top
    ret['search_type'] = 'search_suggest'
    ret['gmt_created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ret['user_id'] = ''
    ret['platformCode'] = ''
    ret['total_time'] = end_time - start_time
    info(json.dumps(ret, ensure_ascii=False))
    return response
