# Image-Segmentation
Build an Image Segmentation System using python

## Using k-means to segment image
Build a 5 dimension feature (r,g,b,x,y), using opencv3.0's k-means to do clustering. 

In main/ directory, run

```
python3 kmeans_segementation.py -i [input-path] -o [output-path] -o2 [output-binary-boundary] -k [cluster-number, default=16]
```

## Using mean shift to segment image
Build a 5 dimension feature (r,g,b,x,y), using sklearn's mean shift to do clustering. Also able to estimate the bandwidth.

In main/ directory, run

```
python3 mean_shift_segmentation.py -i [input-path] -o1 [output-path] -o2 [output-binary-boundary] -k [bandwidth size, default=16]
```

## Evaluation

Here we use the dataset named Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500). It contains the ground truth of segmentation. We compare the predicted boundary with the ground truth boundary by check whether they match in a small patch. The precision is the matched points percentage from the predicted boundary image, while the recall is from the ground truth boundary images. 

After evaluation on 200 images from the BSDS500, the results are shown below:

```
K-means segmentation:
Average precision is: 0.489229963883
Average recall is:0.815504692566

Mean-shift segmentation:
Average precision is: 0.462586406933
Average recall is:0.925816063301
```

We can see that the mean shift method gets better segmentation than k-means. 

## Environment

Python 3.6.2

Python packages:

+ numpy (1.13.3)
+ scipy (1.0.0)
+ scikit-learn (0.19.1)
+ opencv-python (3.3.0.10)


