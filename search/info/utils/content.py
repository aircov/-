import re
from config import es


def _highlight(response_es):
    # 高亮展示es搜索结果

    article_list = response_es["hits"]["hits"]
    for article in article_list:
        source = article["_source"]
        highlight = article["highlight"]
        for key in highlight.keys():
            item = highlight[key][0]
            temp = item
            temp = temp.replace("<em>", "")
            temp = temp.replace("</em>", "")
            item = item.replace("<em>", "<span style='color:red'>")
            item = item.replace("</em>", "</span>")
            source[key] = source[key].replace(temp, item)
    return article_list


def search_content_indistinct(keyword, page, limit):
    if len(keyword) > 1:
        if re.match(r"^[a-zA-Z0-9]+$", keyword):
            # fuzzy “fuzziness”为“编辑距离”,相似度,“prefix_length”前缀相同长度。
            body = {
                "query": {
                    "fuzzy": {
                        "query": {
                            "value": keyword,
                            "fuzziness": 1,  # "fuzziness": "AUTO"
                            "prefix_length": 2
                        }
                    }
                },
                "from": (int(page) - 1) * int(limit),
                "size": int(limit),  # ES默认显示10条数据
            }
        else:
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": keyword,
                                "fields": ["query^2", "source"],  # 在字段末尾添加 ^boost, 代表权重值，默认为1.0
                                "fuzziness": "AUTO",
                                # "operator":"and",  # 多个切词结果在一个item， 等价于 "minimum_should_match":"100%"
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
                "from": (int(page) - 1) * int(limit),
                "size": int(limit),
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
            "from": (int(page) - 1) * int(limit),
            "size": int(limit),
        }

    ret_content = es.search(index="hot_words", doc_type="doc", body=body)
    ret_content = _highlight(ret_content)
    return ret_content


def search_content_exact(keyword, page, limit):
    """
    精确搜索
    :return:
    """
    body = {"query": {"term": {"query.keyword": keyword}}}

    ret_content = es.search(index="hot_words", doc_type="doc", body=body)

    return ret_content
