# local LDA的计算
import numpy as np
import matplotlib.pyplot as plt
from Main.LDA import LDA


def local_lda(data, label):
    """
    计算local LDA
    :param data: local data
    :param label: label of local data
    :return:
    """
    (n, m) = data.shape
    lda = LDA(n_component=m)
    lda.fit_transform(data, label)

    vectors = lda.vectors
    values = lda.values
    return vectors, values


