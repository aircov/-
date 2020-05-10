# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:13
@author : 姚明伟
"""
from flask import Blueprint

# 创建蓝图对象
search_content_blu = Blueprint('search_content', __name__, url_prefix='/api/search/content')

from  .views import *