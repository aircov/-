# -*- coding: utf-8 -*-

import os
import time
import logging
import inspect

dt = time.strftime("%Y%m%d")

handlers = {logging.DEBUG: "./logs/debug_%s.log" % (dt),
            logging.INFO: "./logs/info_%s.log" % (dt),
            logging.WARNING: "./logs/warn_%s.log" % (dt),
            logging.ERROR: "./logs/error_%s.log" % (dt)}
loggers = {}


def init_loggers():
    for level in handlers.keys():
        path = os.path.abspath(handlers[level])
        handlers[level] = logging.FileHandler(path)
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


def info(message):
    message = get_log_msg(message)
    loggers[logging.INFO].info(message)


def error(message):
    message = get_error_msg(message)
    loggers[logging.ERROR].error(message)


def debug(message):
    message = get_log_msg(message)
    loggers[logging.DEBUG].debug(message)


def warn(message):
    message = get_log_msg(message)
    loggers[logging.WARNING].warning(message)
