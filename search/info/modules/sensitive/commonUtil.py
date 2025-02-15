# -*- coding: utf-8 -*-
"""
@time   : 2020/05/10 12:14
@author : 姚明伟
"""
from __future__ import division

import re



def getReg(txt_convert):
    """
    对文本进行正则过滤，检测广告、链接等信息
    :param txt: 文本
    :return: 正则过滤后的文本
    """
    url_patten = r"([^\s]+(\.com))|([a-zA-z]+://[^\s]*)" #http://xxx, www.xxxx.com, 1234@qq.com
    html_patten=r"<(\S*?)[^>]*>.*?|<.*? />"
    # qq_phone_patten=r"[1-9][0-9]{4,}" #第一位1-9之间的数字，第二位0-9之间的数字，大于1000号
    phone=r"1(3|4|5|6|7|8|9)\d{9}$" # 手机号
    wx_patten=r"[a-zA-Z][a-zA-Z0-9_-]{5,19}$"

    if re.findall(url_patten,txt_convert).__len__()>0:
        result = "疑似[网页链接或邮箱]"
    elif re.findall(html_patten,txt_convert).__len__()>0:
        result = "疑似[html脚本]"
    elif re.findall(phone,txt_convert).__len__()>0:
        result = "疑似[手机号]"
    elif re.findall(wx_patten,txt_convert).__len__()>0:
        result = "疑似[微信号]"
    else:
        result ="非广告文本"
    return result



def calcScore(sensitiveWordStr):
    b=sensitiveWordStr
    b1=b.split(",")
    b2=[i.split(":")[0] for i in b1 if len(i) > 1]

    score = 0
    for x in b2:
        if x in ("毒品", "色情", "赌博"):
            score += 5
        elif x in ("政治", "反动", "暴恐"):
            score += 4
        elif x == "社会":
            score += 3
        else: #其他
            score += 2
    return score



def calcGrade(score,sensitive_list_word_length,txt_length):
    if score>15 and sensitive_list_word_length/txt_length>=0.33:
        suggest="删除"
        code = 4003
    elif score==0:
        suggest="通过"
        code = 4000
    else:
        suggest="掩码"
        code = 4002
    return suggest,code
