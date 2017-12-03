import scipy.io
import numpy as np

mat = scipy.io.loadmat('../data/groundTruth/train/2092.mat')

groundTruth = mat.get('groundTruth')
label_num = groundTruth.size

for i in range(label_num):
    boundary = groundTruth[0][i]['Boundaries'][0][0]
    if i == 0:
        trueBoundary = boundary
    else:
        trueBoundary += boundary

print("success")