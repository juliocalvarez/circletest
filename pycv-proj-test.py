# coding=utf-8

# Developed and tested with (fill in):
#
# os:             _?
# python version: _?
# opencv version: _?

# Notes:
#
#    Write here your notes which are important
#  ...
#
#
# Use: something like one of
#    python pycv-proj-test.py images/image_filename
#    python3 pycv-proj-test.py images/image_filename

import numpy as np
import cv2
import matplotlib.pyplot as plt

def euclideanclose(im_bin, count):
    '''
    Escrever texto... bla bla bla....

    :param  im:         The image Input as binary image type.
    :return:            "Binary image with holes filled"
    '''
    # Copy the thresholded image.
    im_in = im_bin.copy()
    notim_in = cv2.bitwise_not(im_in)  # notim_in = 255 - im_in #That's the same thing
    dist_transform = cv2.distanceTransform(notim_in, cv2.DIST_L2, 5)
    ret, dilatation = cv2.threshold(dist_transform, count, 255, cv2.THRESH_BINARY_INV)
    dist_transform2 = cv2.distanceTransform(dilatation.astype('uint8'), cv2.DIST_L2, 5)
    ret2, erosion = cv2.threshold(dist_transform2, count, 255, cv2.THRESH_BINARY)
    im_out = erosion.astype('uint8')
    return im_out

def imfill(im_bin):
    '''
    Escrever texto... bla bla bla....

    :param  im:_bin     The image Input as binary image type.
    :return:            "Binary image with holes filled".
    '''
    # Copy the thresholded image.
    im_in = im_bin.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_bin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_in, mask, (0, 0), 255)
    # Invert floodfilled image
    im_in_not = cv2.bitwise_not(im_in)
    # Combine the two images to get the foreground.
    im_out = im_bin | im_in_not
    #im_bin = im_out
    return im_out

def cleaning(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh, im_bin = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_not = cv2.bitwise_not(im_bin)
    im_close = euclideanclose(im_not, 10)
    edge = cv2.Canny(im_close, 1, 255)
    im_fill = imfill(edge)
    return im_fill

def findcircles(im, im_bin):
    im_copy = im.copy()
    im_bin_copy = im_bin.copy()
    im2, contours, hierarchy = cv2.findContours(im_bin_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_area = []
    # calculate area and filter into new array
    for cnt in contours:
        contours_area.append(cnt)
    # check if contour is of circular shape
    for cnt in contours_area:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            break
        circularity = 4 * 3.14 * (area / (perimeter * perimeter))
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        d = 2 * int(r)  # circles diameter
        if circularity > 0.78 and len(approx) > 11:
            cv2.putText(im_copy, 'circle', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            if d > 10:
                cv2.putText(im_copy, 'd > 10 pixels', (int(x), int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                #cv2.circle(im_copy, (int(x), int(y)), int(r), (255, 0, 0), 2)
            else:
                cv2.putText(im_copy, 'd < 10 pixels', (int(x), int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        else:
            cv2.putText(im_copy, 'non-circle', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
    return im_copy

def draw_circles(im, im_bin):
    im_copy = im.copy()
    rows, cols = im_bin.shape  # Obtain rows and columns
    # Detecting circles in the image with diameter greater than 10 pixels
    circles = cv2.HoughCircles(im_bin, cv2.HOUGH_GRADIENT, 1.134, 50, param1 = 50, param2 = 30, minRadius = 5, maxRadius = 120)
    # ensure at least some circles were found
    if circles is not None:
       print(circles)
       # convert the (x, y) coordinates and radius of the circles to integers
       circles = np.round(circles[0, :]).astype("int")
       # loop over the (x, y) coordinates and radius of the circles
       for (x, y, r) in circles:
           # draw the circle in the output image, then draw a rectangle
           # corresponding to the center of the circle
           cv2.circle(im_copy, (x, y), r, (255, 0, 0), 2)
    return im_copy

def plotting(original, processed):
    # Visualize the result
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})
    ax[0].imshow(original, cmap='gray', interpolation='nearest')
    ax[0].axis('off')
    ax[0].set_title('Original image')
    ax[1].imshow(processed, cmap='gray', interpolation='nearest')
    ax[1].axis('off')
    ax[1].set_title('Processed image')
    fig.tight_layout()
    plt.show()

def main():
    im_original = cv2.imread('./imagens/shapes_leo.jpg')
    #im_original = cv2.imread('./imagens/circles.png')
    im_clean = cleaning(im_original)
    im_circles = findcircles(im_original, im_clean)
    im_processed = draw_circles(im_circles, im_clean)
    plotting(im_original, im_processed)

if __name__=="__main__":
    main()


