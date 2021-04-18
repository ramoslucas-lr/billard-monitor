import cv2 as cv
from scipy.ndimage import maximum_filter, minimum_filter
import numpy as np

cap = cv.VideoCapture('DIP/video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    original = frame.copy()

    gaussian_3 = cv.GaussianBlur(frame, (0, 0), 5.0)
    cv.imshow("gaussian", gaussian_3)
    gaussian_5 = cv.GaussianBlur(gaussian_3, (0, 0), 9.0)
    unsharp_image = cv.addWeighted(gaussian_3, 10, gaussian_5, -9, 0, frame)

    output = unsharp_image.copy()
    b, g, r = cv.split(output)

    circles = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 0.7, 100, param2=20, maxRadius=35)

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
    cv.imshow("output", original)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

#image_url = "DIP/i4.png"
#img = cv.imread(image_url)
#original = cv.imread(image_url)
#cv.imshow('Original', img)

#print("Image Properties")
#print("- Number of Pixels: " + str(img.size))
#print("- Shape/Dimensions: " + str(img.shape))

##gaussian_3 = cv.GaussianBlur(img, (0, 0), 5.0)
##cv.imshow("gaussian", gaussian_3)
##gaussian_5 = cv.GaussianBlur(gaussian_3, (0, 0), 9.0)
##unsharp_image = cv.addWeighted(gaussian_3, 10, gaussian_5, -9, 0, img)

#cv.imshow("unsharpened", unsharp_image)

##output = unsharp_image.copy()
##b,g,r = cv.split(output)
#gray = cv.cvtColor(b, cv.COLOR_BGR2GRAY)
#cv.imshow("b", b)
##circles = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 0.7, 100, param2=20, maxRadius=35)

##if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
##    circles = np.round(circles[0, :]).astype("int")
    # loop over the (x, y) coordinates and radius of the circles
##    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
##        cv.circle(original, (x, y), r, (0, 255, 0), 4)
##        cv.rectangle(original, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image
##    cv.imshow("output", original)
##    cv.waitKey(0)
