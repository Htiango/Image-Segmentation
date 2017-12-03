import scipy.io
import numpy as np


def get_groundTruth(path):
    """
    get the boundary from mat file
    :param path:
    :return:
    """
    mat = scipy.io.loadmat(path)

    groundTruth = mat.get('groundTruth')
    label_num = groundTruth.size

    for i in range(label_num):
        boundary = groundTruth[0][i]['Boundaries'][0][0]
        if i == 0:
            trueBoundary = boundary
        else:
            trueBoundary += boundary

    height = trueBoundary.shape[0]
    width = trueBoundary.shape[1]
    trueBoundary = trueBoundary.reshape(height, width, 1)
    return trueBoundary