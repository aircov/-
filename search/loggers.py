# -*- coding: utf-8 -*-

import os
import time
import logging
import inspect

import multiprocessing
from logging.handlers import TimedRotatingFileHandler
from logging import FileHandler

lock = multiprocessing.Lock()


class SafeLog(TimedRotatingFileHandler):
	def __init__(self, *args, **kwargs):
		super(SafeLog, self).__init__(*args, **kwargs)
		self.suffix_time = ""
		self.origin_basename = self.baseFilename

	def shouldRollover(self, record):
		timeTuple = time.localtime()
		if self.suffix_time != time.strftime(self.suffix, timeTuple) or not os.path.exists(
				self.origin_basename + '.' + self.suffix_time):
			return 1
		else:
			return 0

	def doRollover(self):
		if self.stream:
			self.stream.close()
			self.stream = None

		currentTimeTuple = time.localtime()
		self.suffix_time = time.strftime(self.suffix, currentTimeTuple)
		self.baseFilename = self.origin_basename + '.' + self.suffix_time

		self.mode = 'a'

		global lock
		with lock:
			if self.backupCount > 0:
				for s in self.getFilesToDelete():
					os.remove(s)

		if not self.delay:
			self.stream = self._open()

	def getFilesToDelete(self):
		# 将源代码的 self.baseFilename 改为 self.origin_basename
		dirName, baseName = os.path.split(self.origin_basename)
		fileNames = os.listdir(dirName)
		result = []
		prefix = baseName + "."
		plen = len(prefix)
		for fileName in fileNames:
			if fileName[:plen] == prefix:
				suffix = fileName[plen:]
				if self.extMatch.match(suffix):
					result.append(os.path.join(dirName, fileName))
		if len(result) < self.backupCount:
			result = []
		else:
			result.sort()
			result = result[:len(result) - self.backupCount]
		return result


dt = time.strftime("%Y%m%d")

log_level = 4  # 1:error, 2:warn, 3:info, 4:debug,

handlers = {logging.DEBUG: "./logs/debug.log",
			logging.INFO: "./logs/info.log",
			logging.WARNING: "./logs/warn.log",
			logging.ERROR: "./logs/error.log"}
loggers = {}


def init_loggers():
	for level in handlers.keys():
		path = os.path.abspath(handlers[level])
		# handlers[level] = logging.handlers.TimedRotatingFileHandler(path, when='M', encoding='utf8')
		# SafeLog(handlers[level])
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
