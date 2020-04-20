import json
import re

from flask import request, jsonify

from info.utils.content import search_content_index
from tools.response_code import RET
from . import search_content_blu
from loggers import *


# 关键字搜索内容  GET  http://192.168.191.137:8000/api/search/content/?keyword=%E&page=2&limit=20
@search_content_blu.route('/', methods=['GET'])
def search_content():
    """
    搜内容
    :return:
    """
    keyword = request.args.get('keyword').lower()
    page = request.args.get('page')
    limit = request.args.get('limit')

    # 校验参数
    if not all([keyword, page, limit]):
        warn(json.dumps({'code': RET.PARAMERR, 'errmsg': '参数错误'}, ensure_ascii=False))
        return jsonify(errno=RET.PARAMERR, errmsg='参数不足')

    # es匹配数据
    ret = search_content_index(keyword,page,limit)

    if not ret['hits']['hits']:
        warn(json.dumps({'code': RET.OK, 'errmsg': 'OK', 'keyword':keyword}, ensure_ascii=False))
        return jsonify(errno=RET.OK, errmsg='OK', content=ret['hits']['hits'])

    # 解析json数据,返回给前端
    ret_json = [i['_source'] for i in ret['hits']['hits']]

    search_result = [i.get('id') for i in ret_json]


    result = dict()
    result['code'] = 200
    result['msg'] = "ok"
    result['search_word'] = keyword
    result['search_result'] = search_result
    result['search_type'] = 'content'
    result['gmt_created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # 记录日志
    info(json.dumps(result, ensure_ascii=False))

    return jsonify(errno=RET.OK, errmsg='OK', content=ret_json)
