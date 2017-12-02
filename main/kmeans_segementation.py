import cv2
import numpy as np
import argparse
import time
import find_boundary as bz

def kmeans_seg(img, K):
    size = img.shape
    height = size[0]
    width = size[1]
    channel = size[2]

    imgPos = np.zeros(shape=(height, width, channel + 2))

    for i in range(height):
        for j in range(width):
            imgPos[i][j] = np.append(img[i][j], [i, j])

    Z = imgPos.reshape((-1, 5))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    ret, label, center = cv2.kmeans(Z, K, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten(), 0:3]
    res2 = res.reshape((img.shape))

    mask = bz.find_bound(label, size)

    # mask.dtype = "uint8"

    # cv2.imshow('res2', res2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res2, mask


def seg(args):
    start = time.time()
    img = cv2.imread(args.input_path)
    img_seg, img_binary = kmeans_seg(img, args.K)
    cv2.imwrite(args.output_path, img_seg)
    cv2.imwrite(args.output_path2, img_binary)
    end = time.time()
    print("Spend time: " + str(end - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str,
        required=True, help='path of the input image file')
    parser.add_argument('-o', '--output_path', type=str,
        required=True, help='path of the output image file')
    parser.add_argument('-o2', '--output_path2', type=str,
                        required=True, help='path of the binary output image file')
    parser.add_argument('-k', '--K', type=int,
        default=16, help='the cluster number of kmeans')

    args = parser.parse_args()

    seg(args)

if __name__ == "__main__":
    main()
