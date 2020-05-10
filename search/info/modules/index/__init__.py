# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 19:32
@author : 姚明伟
"""
from flask import Blueprint

# 创建蓝图对象
index_blu = Blueprint('index', __name__)

from  .views import *