import cv2 as cv
import numpy as np
import scipy
import scipy.ndimage

from circles import Circle

image = False
record = False
borders = False

ax = None
ay = None
bx = None
by = None
cx = None
cy = None
dx = None
dy = None

detected_circles = []
shown_circles = []

if not image:
    cap = cv.VideoCapture('DIP/video.mp4')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.

    if record:
        result = cv.VideoWriter('filename.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                30, size)
else:
    image_url = "DIP/i2.png"
    cap = cv.imread(image_url)


def z3_operation(type, src):
    isize = 20
    gray = src.astype('uint16')

    if type == 1:
        Output = scipy.ndimage.grey_dilation(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    if type == 2:
        Output = scipy.ndimage.grey_erosion(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    if type == 3:
        Output = scipy.ndimage.grey_opening(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    if type == 4:
        Output = scipy.ndimage.grey_closing(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))
    if type == 5:
        Output = scipy.ndimage.grey_opening(gray, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        Output = scipy.ndimage.grey_closing(Output, structure=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)))

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

    return Output


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
    double_unsharp_image = cv.addWeighted(gaussian_3, 10, gaussian_5, -9, 0)

    return double_unsharp_image


def unsharp_mask(src):
    gaussian = cv.GaussianBlur(src, (0, 0), 5.0)
    # cv.imshow("gaussian", gaussian)
    unsharp_image = cv.addWeighted(src, 4, gaussian, -3, 0)

    return unsharp_image


def hough_circles(hough):
    circles = cv.HoughCircles(hough, cv.HOUGH_GRADIENT, 0.7, 10,
                              param1=200, param2=15,
                              minRadius=10, maxRadius=20)

    return circles


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_vertices(horizontal, vertical, xm, ym):
    vertical = np.array(vertical)[:, 0]
    horizontal = np.array(horizontal)[:, 1]

    ax = find_nearest(vertical[vertical < xm], xm)
    bx = find_nearest(vertical[vertical > xm], xm)
    cx = bx
    dx = ax

    ay = find_nearest(horizontal[horizontal < ym], ym)
    by = ay
    cy = find_nearest(horizontal[horizontal > ym], ym)
    dy = cy

    return ax, ay, bx, by, cx, cy, dx, dy


def hough_lines(hough):
    height = frame.shape[0]
    width = frame.shape[1]

    xm = width / 2
    ym = height / 2

    vertical_lines = []
    horizontal_lines = []

    # Find the edges in the image using canny detector
    edges = cv.Canny(hough, 50, 200)
    # Detect points that form a line
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=150, maxLineGap=100)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            vertical_lines.append([x1, y1, x2, y2])

        if y1 == y2:
            horizontal_lines.append([x1, y1, x2, y2])

    ax, ay, bx, by, cx, cy, dx, dy = get_vertices(horizontal_lines, vertical_lines, xm, ym)

    return ax, ay, bx, by, cx, cy, dx, dy


def process_frame(frame):
    global borders, ax, ay, bx, by, cx, cy, dx, dy
    original = frame.copy()

    hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    b, g, r = cv.split(original)

    test = z3_operation(2, b)
    cvuint8 = cv.convertScaleAbs(test)
    hough = unsharp_mask(cvuint8)
    cv.imshow('hough', hough)
    circles = hough_circles(hough)

    if not borders:
        ax, ay, bx, by, cx, cy, dx, dy = hough_lines(hough)
        borders = True

    # line ab
    cv.line(original, (ax, ay), (bx, by), (255, 0, 0), 3)
    # line bc
    cv.line(original, (bx, by), (cx, cy), (255, 0, 0), 3)
    # line cd
    cv.line(original, (cx, cy), (dx, dy), (255, 0, 0), 3)
    # line da
    cv.line(original, (ax, ay), (dx, dy), (255, 0, 0), 3)

    shown_circles_arr = []

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        circles_arr = []

        for (x, y, r) in circles:
            if (x > ax - 3 and x < bx + 3) and (y > ay - 3 and y < cy + 3):
                circles_arr.append(Circle(x, y))

        detected_circles.append(circles_arr)


        for circle in circles_arr:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(original, (circle.x, circle.y), 10, (0, 255, 0), 2)
            cv.rectangle(original, (circle.x - 1, circle.y - 1), (circle.x + 1, circle.y + 1), (0, 128, 255), -1)
            shown_circles_arr.append(circle)

        shown_circles.append(shown_circles_arr)




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
            if record:
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
    if record:
        result.release()
    cv.destroyAllWindows()
