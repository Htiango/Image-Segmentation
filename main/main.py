import argparse
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import time
from mean_shift_segmentation import shift_seg
from process_ground_truth import get_groundTruth

def main():

    start = time.time()
    # get all the images files and boundary file
    img_dir = "../data/images/train/"
    boundary_dir = "../data/groundTruth/train/"
    img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith(".jpg")]
    boundary_files = [f for f in listdir(boundary_dir) if isfile(join(boundary_dir, f)) and f.endswith(".mat")]

    num = len(boundary_files)

    for i in range(num):
        print("Handling:   " + boundary_files[i] + "   ====   " + img_files[i])
        img_path = img_files[i]
        boundary_path = boundary_files[i]
        img = cv2.imread(img_path)
        clustered_img, boundary_predict = shift_seg(img)
        boundary_truth = get_groundTruth(boundary_path)
        print("get!")
        break


if __name__ == "__main__":

    main()