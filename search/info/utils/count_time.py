# -*- coding: utf-8 -*-

import time


def count_time(func):
    def int_time(*args, **kwargs):
        start_time = time.time()  # 程序开始时间
        ret = func(*args, **kwargs)
        over_time = time.time()  # 程序结束时间
        total_time = (over_time - start_time)
        print("%s程序共计%.6f秒" % (func, total_time))

        return ret

    return int_time
