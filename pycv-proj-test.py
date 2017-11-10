# coding=utf-8

# Developed and tested with (fill in):
#
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

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def euclideanopen(im_binary, count):
    '''
    Performs erosion followed by dilation. With opening, the original size of the regions is
    essentially retained. Narrow connections between regions and small regions themselves disappear.
    This opening is based on the calculation of Euclidean distance map, thereby preventing the edges of
    objects modified by a structuring element with predefined shape.

    :param  im_binary:      Input image as binary image type
    :param  count:          Number of times erosion and dilation is applied
    :return im_open:        Binary image with spurious elements deleted
    '''
    # Duplicates input image
    im_copy = im_binary.copy()
    # Calculates euclidean distance transform based on the imput image
    dist_transform = cv2.distanceTransform(im_copy, cv2.DIST_L2, 5)
    # Segmentation of the Euclidean distance map which is equivalent to an erosion of size (count) of the input image
    # In this case, the function returns an inverted binary image
    thresh, erosion = cv2.threshold(dist_transform, count, 255, cv2.THRESH_BINARY_INV)
    # Calculates euclidean distance transform based on the inverted erosion image
    dist_transform2 = cv2.distanceTransform(erosion.astype('uint8'), cv2.DIST_L2, 5)
    # Segmentation of the Euclidean distance map of the eroded image, which is equivalent to a
    # dilation of the same size (count) of the input image
    thresh2, dilatation = cv2.threshold(dist_transform2, count, 255, cv2.THRESH_BINARY)
    im_open = dilatation.astype('uint8')
    return im_open

def imfill(im_binary):
    '''
    Fills in the black pixels that are surrounded by a closed outline of white pixels.

    :param  im_binary:     Input image as binary image type
    :return im_fill:       Binary image with holes filled
    '''
    # Duplicates input image
    im_copy = im_binary.copy()
    # Mask used to flood filling. Notice the size needs to be 2 pixels than the image
    h, w = im_binary.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_copy, mask, (0, 0), 255)
    # Inverts floodfilled image
    im_copy_not = cv2.bitwise_not(im_copy) # im_copy_not = 255 - im_copy
    # Combines the two images to get the foreground
    im_fill = im_binary | im_copy_not
    return im_fill

def cleaning(im):
    '''
    Eliminates the spurious elements, returning a binary image where the objects of interest
    are highlighted in white, while image background is black.

    :param  im:         The input image
    :return im_clean:   "Imagem binária limpa com objetos em branco e fundo em preto"
    '''
    # Converts imput image from BGR to GRAY color-space
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Automatic segmentation using Otsu's thresholding
    thresh, im_binary = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Euclidean opening that eliminates the spurious elements of the binary image
    im_open = euclideanopen(im_binary, 10)
    # Detects edges in an image. This function detects relatively thick contours at the edge of bright regions
    edge = cv2.Canny(im_open, 1, 255)
    # Fills the black pixels that are surrounded by a closed outline of white pixels
    im_clean = imfill(edge)
    return im_clean

def contourcircles(im, im_binary, draw, captions):
    '''
    Draws circles, with a diameter greater than 10 pixels, using a function that matches the contours of the
    objects in the binary image. In addition, according to the user's parameters, captions are placed in the images
    on the side of each object showing whether or not it is a circle and, if so, if its diameter is less than or
    greater than 10 pixels.

    :param  im:             First input image
    :param  im_binary:      Second input image as binary image type
    :param  draw:           Indicates the type of circle drawing to be used
    :param  captions:       Indicates whether or not captions will be placed
    :return im_circles:     "Image with caption and/or displayed circles"
    '''
    # Duplicates input images
    im_copy = im.copy()
    im_binary_copy = im_binary.copy()
    # Finds contours in a binary image
    im2, contours, hierarchy = cv2.findContours(im_binary_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Finds all image contours and store them in an array
    contours_array = []
    for cnt in contours:
        contours_array.append(cnt)
    # Loops over contour array elements
    for cnt in contours_array:
        # Calculates perimeter of objects
        perimeter = cv2.arcLength(cnt, True)
        # Calculates area of objects
        area = cv2.contourArea(cnt)
        # Calculates the circularity of objects only in case the perimeter is different from zero.
        # The perimeter can not be zero in the denominator of equation
        if perimeter == 0:
            break
        circularity = 4 * 3.14 * (area / (perimeter * perimeter))
        # Contour Approximation
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Minimum Enclosing Circle
        (x, y), r = cv2.minEnclosingCircle(cnt)
        d = 2 * int(r)  # circles diameter
        # Check if contour is of circular shape
        # Captions are placed on the side of each object showing whether or not it is a circle and, if so, if its diameter is
        # less than or greater than 10 pixels, according to user's specified parameters
        if circularity > 0.78 and len(approx) > 11:
            if captions == 'yes':
                # Draws a text string
                cv2.putText(im_copy, 'circle', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            if d > 10:
                if captions == 'yes':
                    # Draws a text string
                    cv2.putText(im_copy, 'd > 10 pixels', (int(x), int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                if draw == 'contour':
                    # Draw the circle in the output image corresponding to the center of the circle
                    cv2.circle(im_copy, (int(x), int(y)), int(r), (255, 0, 0), 2)
            else:
                if captions == 'yes':
                    # Draws a text string
                    cv2.putText(im_copy, 'd < 10 pixels', (int(x), int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        else:
            if captions == 'yes':
                cv2.putText(im_copy, 'non-circle', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
    im_circles = im_copy
    return im_circles

def houghcircles(im, im_binary):
    '''
    Draws circles, with a diameter greater than 10 pixels, using the Hough Transform.

    :param  im:             Input image
    :param  im_binary:      Input image as binary image type
    :return im_circles:     "Image with drawn circles"
    '''
    im_copy = im.copy()
    # Detects circles in the image with diameter greater than 10 pixels
    circles = cv2.HoughCircles(im_binary, cv2.HOUGH_GRADIENT,
                               1.134, 50, param1 = 50, param2 = 30, minRadius = 5, maxRadius = 120)
    # Ensures at least some circles were found
    if circles is not None:
       # Converts the (x, y) coordinates and radius of the circles to integers
       circles = np.round(circles[0, :]).astype("int")
       # Loops over the (x, y) coordinates and radius of the circles
       for (x, y, r) in circles:
           # Draw the circle in the output image corresponding to the center of the circle
           cv2.circle(im_copy, (x, y), r, (255, 0, 0), 2)
       im_circles = im_copy
       return im_circles

def plotting(original, processed):
    '''
    Create a figure and a set of subplots.

    :param  original:       First input image
    :param  processed:      Second input image
    :return:                "Creates a figure and a set of subplots"
    '''
    # Creates a figure with two subplots
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4), sharex = True, sharey = True,
                           subplot_kw = {'adjustable': 'box-forced'})
    ax[0].imshow(original, cmap = 'gray', interpolation = 'nearest')
    ax[0].axis('off')
    ax[0].set_title('Original image')
    ax[1].imshow(processed, cmap = 'gray', interpolation = 'nearest')
    ax[1].axis('off')
    ax[1].set_title('Processed image')
    fig.tight_layout()
    plt.show()

def main():
    '''
    Starting point of the approach. The method processes user's parameters and then applies developed approach
    for circles detection. As as result, images and solution (processed images) are displayed side by side,
    as required in the problem statement.

    The user can call the method using different arguments, which are explained below
    pycv-proj-test.py [-h] -i IMAGE [-d DRAW] [-c CAPTIONS] [-o OUTPUT]

    optional arguments:
   -h, --help                        show this help message and exit
   -i IMAGE, --image IMAGE           specify the path to the input image
   -d DRAW, --draw DRAW              specify the method to draw circles (Using contours or Hough Transform).
                                     The possible values for this parameter are: contour or hough.
   -c CAPTIONS, --captions CAPTIONS  specify whether or not captions will be placed on image objects
   -o OUTPUT, --output OUTPUT        specify which image will be provided as a result. The possible values for
                                     this parameter are: original or binary.

   Examples:
     python pycv-proj-test.py -i ./imagens/circles.png -d contour -c yes -o binary
        In this example the user will execute the procedure using images/circles.jpg as input image, applying contour
        method, placing caption in image objects and providing the binary image as result
     python pycv-proj-test.py -i ./imagens/shapes_leo.jpg -d hough -c no -o original
        In this example the user will execute the procedure using: images/shapes.png as input image, applying Hough´s
        transform for circles, without placing caption in image objects and providing the original image with circles
        highlighting as result

    :param  im:         List of input parameters defined by user
    :return:            "Binary image with holes filled"
    '''
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required = True,
                    help = 'path to the input image')
    ap.add_argument('-d', '--draw', default = 'hough',
                    help = 'method to draw circles (Using contours or Hough Transform)')
    ap.add_argument('-c', '--captions', default = 'no',
                    help = 'method that places captions on image objects')
    ap.add_argument('-o', '--output', default = 'original',
                    help = 'method to choose the type of output images')
    args = vars(ap.parse_args())

    im_original = cv2.imread(args['image'])
    draw = args['draw']
    captions = args['captions']
    output = args['output']

    im_clean = cleaning(im_original)
    if output == 'binary':
        im_input = im_clean
        im_input = cv2.cvtColor(im_input, cv2.COLOR_GRAY2BGR)
    else:
        im_input = im_original
    im_circles = contourcircles(im_input, im_clean, draw, captions)
    if draw == 'contour':
        im_processed = im_circles
    else:
        im_processed = houghcircles(im_circles, im_clean)
    plotting(im_original, im_processed)

if __name__ == '__main__':
    main()