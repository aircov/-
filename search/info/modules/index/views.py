# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 19:32
@author : 姚明伟
"""
from flask import render_template
from . import index_blu


@index_blu.route('/', methods=['GET'])
def index():
    return render_template('index.html')
