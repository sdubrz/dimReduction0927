# 生成带有heatmap数据的json文件
import numpy as np
import os
import json


def create_json(path="", dim_name=[]):
    """
    生成带有 heatmap 参数的json文件
    要注意参数和参数名的对应关系
    :param path:
    :dim_name: 每个维度属性的名字
    :return:
    """
    heat_path = path + "heatmap\\"

    old_file = open(path+"temp_total.json", encoding='utf-8')
    old_data = json.load(old_file)
    heat_param_file = open(heat_path+"grids.json", encoding='utf-8')
    heat_param = json.load(heat_param_file)

    temp_item = old_data[0]
    dim = temp_item['dNum']
    if len(dim_name)==0:
        for i in range(0, dim):
            dim_name.append("x"+str(i))

    file_writer = open(path+"heatmap.json", 'w', encoding='UTF-8')
    file_writer.write("{\"scatters\": {[")
    for item in old_data:
        file_writer.write(str(item) + "\n")
    file_writer.write("]}, \n")
    file_writer.write("\"heatmap\": {")
    file_writer.write("\"left_low_x\": " + str(heat_param['left_low_x']) + ",\n")
    file_writer.write("\"left_low_y\": " + str(heat_param['left_low_y']) + ",\n")
    file_writer.write("\"rows\": " + str(heat_param['rows']) + ",\n")
    file_writer.write("\"columns\": " + str(heat_param['columns'] + ",\n"))
    file_writer.write("\"r\": " + str(heat_param['r']) + ",\n")
    dim_name_line = "\"dim_name\": " + str(dim_name) + ",\n"
    # 还差各个heatmap没写

    file_writer.write("}\n")
    file_writer.write("}")

    file_writer.close()


if __name__ == '__main__':
    path = "E:\\Project\\result2019\\result1026without_straighten\\PCA\\Wine\\yita(0.05)nbrs_k(40)method_k(40)numbers(4)_b-spline_weighted\\"
    create_json(path)
