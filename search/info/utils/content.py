import re
from config import es



def search_content_index(keyword, page, limit):
    if len(keyword) > 1:
        if re.match(r'^[a-zA-Z0-9]+$', keyword):
            # fuzzy “fuzziness”为“编辑距离”,相似度,“prefix_length”前缀相同长度。
            body = {

                "query": {
                    "bool": {
                        "must":{
                            "bool":{
                                "should": [
                                    {
                                        "fuzzy": {
                                            "title": {
                                                "value": keyword,
                                                "fuzziness": 1,
                                                "prefix_length": 2
                                            }
                                        }

                                    },
                                    {
                                        "fuzzy": {
                                            "content": {
                                                "value": keyword,
                                                "fuzziness": 1,
                                                "prefix_length": 2
                                            }
                                        }
                                    }
                                ]
                            }

                        },
                        "filter": {
                            "bool":{
                                "must": [
                                    {"term": {"deleted": "0"}},
                                    {"term": {"status": "published"}}
                                ],
                            }
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
                                'fields': ['title', 'content', 'longContent'],
                                'fuzziness': 'AUTO',
                            }
                        },
                        "filter": {
                            "bool":{
                                "must": [
                                    {"term": {"deleted": "0"}},
                                    {"term": {"status": "published"}}
                                ],
                            }
                        }
                    }
                },
                'from': (int(page) - 1) * int(limit),
                'size': int(limit)
            }

    else:
        # 单个字搜索
        body = {
            "query": {

                "bool": {
                    "must": {
                        "bool": {
                            "should": [{
                                "wildcard": {
                                    "title": "*" + keyword + "*",
                                }
                            }, {
                                "wildcard": {
                                    "content": "*" + keyword + "*",
                                }
                            }, {
                                "wildcard": {
                                    "longContent": "*" + keyword + "*",
                                }
                            }]
                        }
                    },
                    "filter": {
                        "bool":{
                            "must": [
                                {"term": {"deleted": "0"}},
                                {"term": {"status": "published"}}
                            ],
                        }
                    }
                }
            },
            'from': (int(page) - 1) * int(limit),
            'size': int(limit),
        }

    ret_content = es.search(index='subject', doc_type='wbc_subject', body=body)
    return ret_content
