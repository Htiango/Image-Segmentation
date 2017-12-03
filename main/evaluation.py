import argparse
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, splitext
import time
from mean_shift_segmentation import shift_seg
from process_ground_truth import get_groundTruth
from eval_boundary import eval_bound
from kmeans_segementation import kmeans_seg


def main():
    OUTPUT_PATH_KMEANS = '../output/kmeans-seg/'
    OUTPUT_PATH_SHIFT = '../output/meanshift-seg/'
    OUTPUT_PATH_TRUTH = '../output/truth-seg/'

    K_CLUSTER = 16

    OUTPUT_LOG = '../output/log.txt'

    f = open(OUTPUT_LOG, "w+")

    start = time.time()
    # get all the images files and boundary file
    img_dir = "../data/images/train/"
    boundary_dir = "../data/groundTruth/train/"
    img_files = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith(".jpg")]
    boundary_files = [f for f in listdir(boundary_dir) if isfile(join(boundary_dir, f)) and f.endswith(".mat")]

    f.write("Get a list of images and boundary file\n")

    num = len(boundary_files)

    f.write("Image num is: " + str(num) + "\n\n")

    total_precision = 0.0
    total_recall = 0.0

    for i in range(num):
        start_iteration = time.time()
        print("Handling:   " + boundary_files[i] + "   ====   " + img_files[i])
        f.write("Handling:   " + boundary_files[i] + "   ====   " + img_files[i] + "\n")

        image_name = img_files[i]
        name, extension = splitext(image_name)

        img_path = join(img_dir, image_name)

        boundary_path = join(boundary_dir, boundary_files[i])
        img = cv2.imread(img_path)

        start_kmeans = time.time()
        clustered_img_kmeans, boundary_predict_kmeans = kmeans_seg(img, K_CLUSTER)
        end_kmeans = time.time()
        print("Kmeans time: " + str(end_kmeans - start_kmeans))
        f.write("Kmeans time: " + str(end_kmeans - start_kmeans) + "\n")

        start_shift = time.time()
        clustered_img_shift, boundary_predict_shift = shift_seg(img)
        end_shift = time.time()
        print("Meanshift time: " + str(end_shift - start_shift))
        f.write("Meanshift time: " + str(end_shift - start_shift) + "\n")

        output_kmeans_path_seg = OUTPUT_PATH_KMEANS + name + '-seg.jpg'
        output_kmeans_path_binary = OUTPUT_PATH_KMEANS + name + '-boundary-kmeans.png'

        output_shift_path_seg = OUTPUT_PATH_SHIFT + name + '-seg.jpg'
        output_shift_path_binary = OUTPUT_PATH_SHIFT + name + '-boundary-shift.png'

        cv2.imwrite(output_kmeans_path_seg, clustered_img_kmeans)
        cv2.imwrite(output_kmeans_path_binary, boundary_predict_kmeans)

        cv2.imwrite(output_shift_path_seg, clustered_img_shift)
        cv2.imwrite(output_shift_path_binary, boundary_predict_shift)

        boundary_truth = get_groundTruth(boundary_path)

        output_truth_path_binary = OUTPUT_PATH_TRUTH + name + '-boundary-truth.png'

        cv2.imwrite(output_truth_path_binary, boundary_truth)

        precision, recall = eval_bound(boundary_predict_kmeans, boundary_truth, 4)
        end_iteration = time.time()
        print("Precision is: " + str(precision))
        print("Recall is:" + str(recall))

        f.write("Precision is: " + str(precision) + "\n")
        f.write("Recall is:" + str(recall) + "\n")

        total_precision += precision
        total_recall += recall

        print("Spend time: " + str(end_iteration - start_iteration))
        f.write("Spend time: " + str(end_iteration - start_iteration) + "\n\n")
        # break

    average_precision = total_precision / num
    average_recall = total_recall / num
    print("Average precision is: " + str(average_precision))
    print("Average recall is:" + str(average_recall))
    f.write("Average precision is: " + str(average_precision) + "\n")
    f.write("Average recall is:" + str(average_recall) + "\n")
    f.close()


def evaluate_from_image():
    OUTPUT_PATH_KMEANS = '../output/kmeans-seg/'
    OUTPUT_PATH_SHIFT = '../output/meanshift-seg/'
    OUTPUT_PATH_TRUTH = '../output/truth-seg/'

    PATH = OUTPUT_PATH_KMEANS

    OUTPUT_LOG = '../output/log.txt'

    f = open(OUTPUT_LOG, "w+")

    start = time.time()
    # get all the boundary file
    boundary_dir = "../data/groundTruth/train/"
    boundary_files = [f for f in listdir(boundary_dir) if isfile(join(boundary_dir, f)) and f.endswith(".mat")]

    f.write("Get a list of images and boundary file\n")

    num = len(boundary_files)

    f.write("Image num is: " + str(num) + "\n\n")

    total_precision = 0.0
    total_recall = 0.0

    for i in range(num):
        start_iteration = time.time()
        print("Handling:   " + boundary_files[i] )
        f.write("Handling:   " + boundary_files[i] + "\n")

        name, extension = splitext(boundary_files[i])

        # img_path = join(PATH, name+'-boundary-shift.png')

        img_path = join(PATH, name + '-boundary-kmeans.png')

        boundary_path = join(boundary_dir, boundary_files[i])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = img.shape
        boundary_predict_shift = img.reshape(size[0], size[1], 1)

        boundary_truth = get_groundTruth(boundary_path)

        precision, recall = eval_bound(boundary_predict_shift, boundary_truth, 4)
        print("Precision is: " + str(precision))
        print("Recall is:" + str(recall))

        f.write("Precision is: " + str(precision) + "\n")
        f.write("Recall is:" + str(recall) + "\n\n")

        total_precision += precision
        total_recall += recall

    average_precision = total_precision / num
    average_recall = total_recall / num
    print("Average precision is: " + str(average_precision))
    print("Average recall is:" + str(average_recall))
    f.write("Average precision is: " + str(average_precision) + "\n")
    f.write("Average recall is:" + str(average_recall) + "\n")
    f.close()


if __name__ == "__main__":
    evaluate_from_image()
