import json
import re
import pandas as pd

from flask import request, jsonify, make_response

from info.utils.content import search_content_indistinct, search_content_exact
from info.utils.thread_return_value import MyThread
from loggers import *

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
    # start_time = time.time()
    keyword = request.args.get('keyword')
    page = request.args.get('page')
    limit = request.args.get('limit')
    # print(request.remote_addr)
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
    ret['indistinct_people'] = indistinct_people['hits']['hits']
    ret['exact_people'] = exact_people.get('hits').get('hits') if exact_people.get('hits').get('hits') else []

    return jsonify(code=RET.OK, message='OK', body=ret)


# 构造字典树
from info.modules.search_person import search_script_conf


@search_content_blu.route('/su/')
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
