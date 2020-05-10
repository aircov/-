# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:13
@author : 姚明伟
"""
from flask import Blueprint

# 创建蓝图对象
sensitive_blu = Blueprint('sensitive', __name__, url_prefix='/api')

from  .views import *