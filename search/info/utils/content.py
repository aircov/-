import re
from config import es



def search_content_indistinct(keyword, page, limit):
    if len(keyword) > 1:
        if re.match(r'^[a-zA-Z0-9]+$', keyword):
            # fuzzy “fuzziness”为“编辑距离”,相似度,“prefix_length”前缀相同长度。
            body = {
                "query": {
                    "fuzzy": {
                        "query": {
                            "value": keyword,
                            "fuzziness": 1,  # 'fuzziness': 'AUTO'
                            "prefix_length": 2
                        }
                    }
                },
                'from': (int(page) - 1) * int(limit),
                'size': int(limit),  # ES默认显示10条数据
            }
        else:
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                'query': keyword,
                                'fields': ['query'],
                                'fuzziness': 'AUTO',
                            }
                        },
                        # "filter": {
                        #     "bool":{
                        #         "must": [
                        #             {"term": {"deleted": "0"}},
                        #             {"term": {"status": "published"}}
                        #         ],
                        #     }
                        # }
                    }
                },
                'from': (int(page) - 1) * int(limit),
                'size': int(limit),
                # 排序
                # "sort": {
                #     "update_time": {
                #         "order": "desc"
                #     }
                # },
                # 高亮
                "highlight": {
                    "fields": {
                        "query": {}
                    }
                }
            }

    else:
        # 单个字搜索
        body = {
            "query": {
                "wildcard": {
                    "query": "*" + keyword + "*",
                }
            },
            'from': (int(page) - 1) * int(limit),
            'size': int(limit),
        }

    ret_content = es.search(index='hot_words', doc_type='doc', body=body)
    return ret_content

def search_content_exact(keyword, page, limit):
    """
    精确搜索
    :return:
    """
    body = {"query": {"term": {"query.keyword": keyword}}}

    ret_content = es.search(index='hot_words', doc_type='doc', body=body)
    return ret_content