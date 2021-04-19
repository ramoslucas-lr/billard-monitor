import math

import cv2 as cv
import numpy as np
import scipy
import scipy.ndimage
from matplotlib import pyplot as plt

image = False

if not image:
    cap = cv.VideoCapture('DIP/video.mp4')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv.VideoWriter('filename.avi',
                             cv.VideoWriter_fourcc(*'MJPG'),
                             10, size)
else:
    image_url = "DIP/i4.png"
    cap = cv.imread(image_url)


def z3_operation(type, src):
    isize = 20
    gray = src.astype('uint16')

    # Output1 = scipy.ndimage.grey_dilation(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    Output2 = scipy.ndimage.grey_erosion(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4)))
    # Output3 = scipy.ndimage.grey_opening(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    # Output4 = scipy.ndimage.grey_closing(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    # Output5 = scipy.ndimage.grey_closing(Output3, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))

    # f, axarr = plt.subplots(3,2)
    # axarr[0, 0].imshow(gray, cmap='gray')
    # axarr[0, 0].text(0, 0, str('Input Image'))
    # axarr[0, 1].imshow(Output1, cmap='gray')
    # axarr[0, 1].text(0, 0, str('Dilated Image'))
    # axarr[1, 0].imshow(Output2, cmap='gray')
    # axarr[1, 0].text(0, 0, str('Eroded Image'))
    # axarr[1, 1].imshow(Output3, cmap='gray')
    # axarr[1, 1].text(0, 0, str('Opened Image'))
    # axarr[2, 0].imshow(Output4, cmap='gray')
    # axarr[2, 0].text(0, 0, str('Closed Image'))
    # axarr[2, 1].imshow(Output5, cmap='gray')
    # axarr[2, 1].text(0, 0, str('Opened and Closed Image'))
    #
    # plt.show()

    return Output2


def threshold_image(src):
    reto, th1 = cv.threshold(src, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    return th1, th2, th3


def local_histogram_equalization(src):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(90, 90))
    cl1 = clahe.apply(src)

    return cl1


def double_unsharp_mask(src):
    gaussian_3 = cv.GaussianBlur(src, (0, 0), 5.0)
    gaussian_5 = cv.GaussianBlur(gaussian_3, (0, 0), 9.0)
    double_unsharp_image = cv.addWeighted(gaussian_3, 10, gaussian_5, -9)

    return double_unsharp_image


def unsharp_mask(src):
    gaussian = cv.GaussianBlur(src, (0, 0), 5.0)
    # cv.imshow("gaussian", gaussian)
    unsharp_image = cv.addWeighted(src, 2, gaussian, -1, 0)

    return unsharp_image


def process_frame(frame):
    original = frame.copy()

    b, g, r = cv.split(original)

    test = z3_operation('', b)
    cvuint8 = cv.convertScaleAbs(test)
    hough = unsharp_mask(cvuint8)
    # cv.imshow('hough', hough)
    circles = cv.HoughCircles(hough, cv.HOUGH_GRADIENT, 0.1, 10,
                              param1=180, param2=15,
                              minRadius=10, maxRadius=20)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(original, (x, y), r, (0, 255, 0), 4)
            cv.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image

    return original


if image:
    output = process_frame(cap)
    cv.imshow("output", output)
    cv.waitKey(0)

else:
    while cap.isOpened():

        ret, frame = cap.read()

        if ret == True:

            output = process_frame(frame)
            # Write the frame into the
            # file 'filename.avi'
            result.write(output)

            # Display the frame
            # saved in the file
            cv.imshow('Frame', output)

            # Press S on keyboard
            # to stop the process
            if cv.waitKey(1) & 0xFF == ord('s'):
                break

            # Break the loop
        else:
            break



    cap.release()
    result.release()
    cv.destroyAllWindows()
