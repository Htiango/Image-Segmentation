import argparse
import cv2
import numpy as np


def readImg():
    # path = args.input_image
    path = "../data/Polarlicht_2.jpg"
    img = cv2.imread(path)
    size = img.shape
    height = size[0]
    width = size[1]
    channel = size[2]

    imgPos = np.zeros(shape=(height,width,channel + 2))

    for i in range(height):
        for j in range(width):
            imgPos[i][j] = np.append(img[i][j], [i, j])

    Z = imgPos.reshape((-1, 5))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten(), 0:3]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', type=str,
        required=True, help='path of the input image file')
    args = parser.parse_args()

    # readimg(args)

if __name__ == "__main__":
    readImg()