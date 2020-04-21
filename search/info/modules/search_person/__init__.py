from flask import Blueprint

# 创建蓝图对象
search_person_blu = Blueprint('search_person', __name__, url_prefix='/api/search/person')

from  .views import *