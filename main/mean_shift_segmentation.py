import cv2
import numpy as np
import argparse
import time
import copy
from find_boundary import find_bound
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

def shift_seg(img, K):
    size = img.shape
    height = size[0]
    width = size[1]
    channel = size[2]

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')




    #img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    imgPos = np.zeros(shape=(height, width, channel + 2))

    for i in range(height):
        for j in range(width):
            imgPos[i][j] = np.append(img[i][j], [i, j])

    Z1 = imgPos.reshape((-1, 5))

    Z2 = img.reshape((-1, 3))
    # convert to np.float32
    Z1 = np.float32(Z1)
    Z2 = np.float32(Z2)



    #bandwidth = 30
    bandwidth = estimate_bandwidth(Z2, quantile=0.2, n_samples=1000)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Z1)
    label1 = ms.labels_

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(Z2)
    label2 = ms.labels_


    plt.subplot(3, 1, 2)
    plt.imshow(np.reshape(label1, img.shape[0:2]))
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(np.reshape(label2, img.shape[0:2]))
    plt.axis('off')
    plt.show()



    # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # res = center[label.flatten(), 0:3]
    # res2 = res.reshape((img.shape))
    #
    # mask = find_bound(label,size)

    # cv2.imshow('res2', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return res2
    return 0


def seg(args):
    start = time.time()
    img = cv2.imread(args.input_path)
    img_seg = shift_seg(img, args.K)
    #cv2.imwrite(args.output_path, img_seg)
    end = time.time()
    print("Spend time: " + str(end - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str,
        required=True, help='path of the input image file')
    parser.add_argument('-o', '--output_path', type=str,
        required=True, help='path of the ouput image file')
    parser.add_argument('-k', '--K', type=int,
        default=16, help='bandwidth')

    args = parser.parse_args()

    seg(args)

if __name__ == "__main__":
    main()
