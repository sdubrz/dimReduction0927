# 用 Dijstra 计算单源最短路径


class Edge:
    def __init__(self, v1, v2, weight):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight

    def equals(self, edge2):
        """判断两个边是否相等"""
        if self.v1 == edge2.v1 and self.v2 == edge2.v2 and self.weight == edge2.weight:
            return True
        else:
            return False

    def to_string(self):
        s = "(" + str(self.v1) + ", " + str(self.v2) + ", " + str(self.weight)+")"


class Graph:
    def __init__(self, v_num):
        self.v_num = v_num
        self.edge_list = []

