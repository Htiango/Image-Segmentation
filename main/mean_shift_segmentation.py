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

    # plt.figure(2)
    # plt.subplot(3, 1, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    imgPos = np.zeros(shape=(height, width, channel + 2))

    for i in range(height):
        for j in range(width):
            imgPos[i][j] = np.append(img[i][j], [i, j])

    Z_feature5 = imgPos.reshape((-1, 5))
    Z_feature3 = img.reshape((-1, 3))
    Z_feature5 = np.float32(Z_feature5)
    Z_feature3 = np.float32(Z_feature3)

    # bandwidth = 30
    bandwidth = estimate_bandwidth(Z_feature3, quantile=0.2, n_samples=1000)
    print(bandwidth)
    ms_feature5 = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms_feature5.fit(Z_feature5)
    label_feature5 = ms_feature5.labels_
    center_feature5 = np.uint8(ms_feature5.cluster_centers_)
    res_feature5 = center_feature5[label_feature5.flatten(), 0:3]
    res_feature5 = res_feature5.reshape(img.shape)
    # res_feature5 = cv2.cvtColor(res_feature5, cv2.COLOR_YUV2BGR)
    res_feature5 = cv2.cvtColor(res_feature5, cv2.COLOR_LAB2BGR)

    # ms_feature3 = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms_feature3.fit(Z_feature3)
    # label_feature3 = ms_feature3.labels_
    # center_feature3 = np.uint8(ms_feature3.cluster_centers_)
    # res_feature3 = center_feature3[label_feature3.flatten(), 0:3]
    # res_feature3 = res_feature3.reshape(img.shape)

    # plt.subplot(3, 1, 2)
    # plt.imshow(res_feature5)
    # plt.axis('off')
    # plt.subplot(3, 1, 3)
    # plt.imshow(res_feature3)
    # plt.axis('off')
    # plt.show()

    # mask_feature3 = find_bound(label_feature3, size)
    mask_feature5 = find_bound(label_feature5, size)

    return res_feature5, mask_feature5


def seg(args):
    start = time.time()
    img = cv2.imread(args.input_path)
    img_seg5, mask5 = shift_seg(img, args.K)
    cv2.imwrite(args.output_path, img_seg5)
    cv2.imwrite(args.output_path2, mask5)
    end = time.time()
    print("Spend time: " + str(end - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str,
        required=True, help='path of the input image file')
    parser.add_argument('-o1', '--output_path', type=str,
        required=True, help='path of the ouput image file')
    parser.add_argument('-o2', '--output_path2', type=str,
                        required=True, help='path of the binary output image file')
    parser.add_argument('-k', '--K', type=int,
        default=16, help='bandwidth')

    args = parser.parse_args()

    seg(args)

if __name__ == "__main__":
    main()
