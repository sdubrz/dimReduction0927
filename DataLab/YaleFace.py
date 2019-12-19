import numpy as np


def see_data():
    path = "E:\\Project\\DataLab\\YaleFace\\"
    data = np.load(path+"subject01.centerlight", allow_pickle=True)
    print(data.shape)


if __name__ == '__main__':
    see_data()
