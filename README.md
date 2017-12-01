# Image-Segmentation
Build an Image Segmentation System using python


## Using k-means to segment image
Build a 5 dimension feature (r,g,b,x,y), using opencv3.0's k-means to do clustering. 

In main/ directory, run

```
python3 kmeans_segementation.py -i [input-path] -o [output-path] -k [cluster-number, default=16]
```

