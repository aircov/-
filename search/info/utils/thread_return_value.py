"""
自定义多线程类,获取子线程返回值
"""
import threading


class MyThread(threading.Thread):
    def __init__(self, target=None, args=(), **kwargs):
        super(MyThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        if self._target == None:
            return
        self.__result__ = self._target(*self._args, **self._kwargs)

    def get_result(self):
        # 当需要取得结果值的时候阻塞等待子线程完成
        self.join()

        return self.__result__
