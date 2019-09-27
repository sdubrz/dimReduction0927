import numpy as np


"""
    对正负扰动不成直线的情况进行校直
    
    @author: sdu_brz
    @date: 2019/03/04
"""


def straightening(y0, y_add, y_sub):
    """
    对正负扰动不共线的情况进行校直
    :param y0: 没有扰动的降维结果
    :param y_add: 正向扰动的降维结果
    :param y_sub: 负向扰动的降维结果
    :return:
    """
    v1 = y_add - y0
    v2 = y_sub - y0

    y1 = 0.5 * (v1 - v2)
    y2 = 0.5 * (v2 - v1)

    return y1, y2
