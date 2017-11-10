
# Test for Python + OpenCV developers

# author:         Julio César Álvarez Iglesias
# os:             Windows 7
# python version: 3.5.2
# opencv version: 3.1.0
# tested with:    two images provided shapes_leo.jpg and circles.png

# Notes:
# Routine that detects circles with diameter higher than 10 pixels. It is important to note that
# the developed routing follows the same approach for both images provided. Additionally, it provides
# flexibility by allowing the user to use different parameters configuration (this is detailed in step 4 below)
#
# The approach encompasses the following steps:
#
# 1- The image was segmented using automatic thresholding method (Otsu method).
# Once the image has been binarized, it is filtered, eliminating spurious elements. In order to do that,
# the Euclidean opening morphological function was used. This opening is based on the
# map of Euclidean distances and has the advantage of not using structuring element. Hence,
# the objects edge become smooth instead of being modified by the structuring element.
#
# 2- Once the spurious elements were eliminated, it was applied an edge detection function (Canny)
# followed by a fill function in order to highlight the objects of interest in white, while the image background
# is black.
#
# 3- In this step the image is processed in order to identify circles with
# diameter greater than 10 pixels. Two methods are used: (i) Hough´s transform for circles and (ii) a function
# that finds the contours of objects.
#
# 4- Finally, the original image and solution (processed image) are displayed side by side, as required in the
# problem statement.
#
# Additionally, several parameterizations were used in order to provide the user with more flexibility regarding
# solution options and visualization of results, as described below:
#
# pycv-proj-test.py [-h] -i IMAGE [-d DRAW] [-c CAPTIONS] [-o OUTPUT]
#
# optional arguments:
#   -h, --help                        show this help message and exit
#   -i IMAGE, --image IMAGE           specify the path to the input image
#   -d DRAW, --draw DRAW              specify the method to draw circles (Using contours or Hough Transform). The
#                                     possible values for this parameter are: contour or hough. The first applied
#                                     contour approach, while the later uses Hough´s transform for circles, respectivelly
#   -c CAPTIONS, --captions CAPTIONS  specify whether or not captions will be placed on image objects
#   -o OUTPUT, --output OUTPUT        specify which image will be provided as result. The possible values for
#                                     this parameter are: original or binary.
#
# Examples:
#   python pycv-proj-test.py -i ./imagens/circles.png -d contour -c yes -o binary
#      In this example the user will execute the procedure using images/circles.png as input image, applying contour
#      method, placing caption in image objects and providing the binary image as result
#   python pycv-proj-test.py -i ./imagens/shapes_leo.jpg -d hough -c no -o original
#      In this example the user will execute the procedure using: images/shapes_leo.jpg as input image, applying
#      Hough´s transform for circles, without placing caption in image objects and providing the original image with
#      circles highlighting as result