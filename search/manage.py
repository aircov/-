# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:13
@author : 姚明伟
"""
from flask_script import Manager


from info import create_app

app = create_app('development')  # 开发
# app = create_app('production')  # 上线

manager = Manager(app)


if __name__ == '__main__':
    manager.run()
