import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def test():
    path = "C:\\Users\\brz\\Downloads\\Marcel-Train\\A\\"
    a = Image.open(path+"A-train0004.ppm")
    a.show()
    B = np.array(a)
    print(B.shape)


if __name__ == '__main__':
    test()

