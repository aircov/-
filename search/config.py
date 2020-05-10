import logging
import redis
import pymysql
from elasticsearch import Elasticsearch


class Config(object):
    """项目配置"""
    DEBUG = True


class DevelopmentConfig(Config):
    """开发环境下的配置"""
    DEBUG = True
    LOG_LEVEL = logging.INFO


class ProductConfig(Config):
    "生产环境下的配置"
    DEBUG = False
    LOG_LEVEL = logging.WARNING


config = {
    "development": DevelopmentConfig,
    "production": ProductConfig,
    # "testing": TestingConfig
}


# 线上
conn_redis = redis.StrictRedis('127.0.0.1',6379,0,decode_responses=True)

es = Elasticsearch(host="127.0.0.1")

