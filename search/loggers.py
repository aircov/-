# -*- coding: utf-8 -*-

import os
import time
import logging
import inspect

from SafeLog import SafeLog

dt = time.strftime("%Y%m%d")

log_level = 4   # 1:error, 2:warn, 3:info, 4:debug,

handlers = {logging.DEBUG: "./logs/debug.log",
            logging.INFO: "./logs/info.log",
            logging.WARNING: "./logs/warn.log",
            logging.ERROR: "./logs/error.log"}
loggers = {}


def init_loggers():
    for level in handlers.keys():
        path = os.path.abspath(handlers[level])
        #handlers[level] = logging.handlers.TimedRotatingFileHandler(path, when='M', encoding='utf8')
        #SafeLog(handlers[level])
        handlers[level] = SafeLog(path, when='midnight', encoding='utf8')
        # handlers[level] = logging.handlers.TimedRotatingFileHandler(path, when='midnight', encoding='utf8')
    for level in handlers.keys():
        logger = logging.getLogger(str(level))
        # 如果不指定level，获得的handler似乎是同一个handler
        logger.addHandler(handlers[level])
        logger.setLevel(level)
        loggers.update({level: logger})


# 加载模块时创建全局变量
init_loggers()


def print_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def get_log_msg(message):
    return "[%s]  %s" % (print_now(), message)


def get_error_msg(message):
    frame, filename, lineNo, functionName, code, unknowField = inspect.stack()[2]
    return "[%s] [%s - %s - %s] %s" % (print_now(), filename, lineNo, functionName, message)


def debug(message):
    if log_level > 3:
        message = get_log_msg(message)
        loggers[logging.DEBUG].debug(message)


def info(message):
    if log_level > 2:
        message = get_log_msg(message)
        loggers[logging.INFO].info(message)


def warn(message):
    if log_level > 1:
        message = get_log_msg(message)
        loggers[logging.WARNING].warning(message)


def error(message):
    message = get_error_msg(message)
    loggers[logging.ERROR].error(message)

