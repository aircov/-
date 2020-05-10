# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:13
@author : 姚明伟
"""
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask

from config import config


def setup_log(config_name):
    # 设置日志的记录等级
    logging.basicConfig(level=config[config_name].LOG_LEVEL)  # 调试debug级
    # 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
    file_log_handler = RotatingFileHandler("logs/info.log", maxBytes=1024 * 1024 * 100, backupCount=10)
    # 创建日志记录的格式 日志等级 输入日志信息的文件名 行数 日志信息
    formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d %(message)s')
    # 为刚创建的日志记录器设置日志记录格式
    file_log_handler.setFormatter(formatter)
    # 为全局的日志工具对象（flask app使用的）添加日志记录器
    logging.getLogger().addHandler(file_log_handler)


def create_app(config_name):
    # 配置日志,并且传入配置名字，以便能获取到指定配置所对应的日志等级
    setup_log(config_name)
    app = Flask(__name__)
    # 加载配置
    app.config.from_object(config[config_name])

    # 注册蓝图
    from info.modules.search_content import search_content_blu
    app.register_blueprint(search_content_blu)
    from info.modules.sensitive import sensitive_blu
    app.register_blueprint(sensitive_blu)
    from info.modules.index import index_blu
    app.register_blueprint(index_blu)

    return app
