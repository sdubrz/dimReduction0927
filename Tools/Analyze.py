import numpy as np
import matplotlib.pyplot as plt

def heatmap_test():
    path = "D:\\Exp\\cTSNE\\abalone500\\yita(0.20200213)nbrs_k(10)method_k(30)numbers(5)_b-spline_weighted\\"
    X = np.loadtxt(path+"cTSNE_Pxy.csv", dtype=np.float, delimiter=",")
    plt.imshow(X, cmap="spring")
    plt.show()

if __name__ == '__main__':
    heatmap_test()