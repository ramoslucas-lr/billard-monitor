import cv2 as cv
import numpy as np
import scipy
import scipy.ndimage
import csv
import joblib

from matplotlib import colors

borders = False

border_coordinates = {'ax': 0, 'ay': 0, 'bx': 0, 'by': 0,
                      'cx': 0, 'cy': 0, 'dx': 0, 'dy': 0}

frame_width = 0
frame_height = 0
size = (0, 0)

ball_classes = ['pink', 'yellow', 'trash', 'yellow', 'brown', 'trash', 'red', 'black', 'green', 'white', 'blue']

def load_model(filename):
    multi_model = joblib.load(filename)
    return multi_model

class Circle:
    def __init__(self, x_coord, y_coord, radius):
        self.x = x_coord
        self.y = y_coord
        self.r = radius


def z3_operation(operation, src):
    if operation == 1:
        return cv.dilate(src, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    elif operation == 2:
        return cv.erode(src, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    elif operation == 3:
        return cv.morphologyEx(src, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    elif operation == 4:
        return cv.morphologyEx(src, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    elif operation == 5:
        opened = cv.morphologyEx(src, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        return cv.morphologyEx(opened, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    return src


def unsharp_mask(src):
    gaussian = cv.GaussianBlur(src, (0, 0), 5.0)
    unsharp_image = cv.addWeighted(src, 4, gaussian, -3, 0)

    return unsharp_image


def hough_circles(hough):
    circles = cv.HoughCircles(hough, cv.HOUGH_GRADIENT, 0.7, 10, param1=200, param2=15, minRadius=10, maxRadius=20)
    return circles


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_vertices(horizontal, vertical, xm, ym):
    global border_coordinates
    vertical = np.array(vertical)[:, 0]
    horizontal = np.array(horizontal)[:, 1]

    border_coordinates['ax'] = find_nearest(vertical[vertical < xm], xm)
    border_coordinates['dx'] = border_coordinates['ax']
    border_coordinates['bx'] = find_nearest(vertical[vertical > xm], xm)
    border_coordinates['cx'] = border_coordinates['bx']

    border_coordinates['ay'] = find_nearest(horizontal[horizontal < ym], ym)
    border_coordinates['by'] = border_coordinates['ay']
    border_coordinates['cy'] = find_nearest(horizontal[horizontal > ym], ym)
    border_coordinates['dy'] = border_coordinates['cy']


def hough_lines(hough):
    """
    D --------------------------------- C
    |                                   |
    |      Table borders are formed     |
    |         by AB, BC, CD and DA      |
    |              lines                |
    |                                   |
    A --------------------------------- B

    :param hough:
        Eroded blue channel passed through an unsharp masking filter
    """
    global frame_width, frame_height

    xm = frame_width / 2
    ym = frame_height / 2

    vertical_lines = []
    horizontal_lines = []

    edges = cv.Canny(hough, 50, 200)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=150, maxLineGap=100)

    for line in lines:
        x1, y1, x2, y2 = line.flatten()[:4]

        if x1 == x2:
            vertical_lines.append([x1, y1, x2, y2])

        if y1 == y2:
            horizontal_lines.append([x1, y1, x2, y2])

    get_vertices(horizontal_lines, vertical_lines, xm, ym)


def preprocess(frame):
    b, g, r = cv.split(frame)

    eroded = z3_operation(2, b)
    cvuint8 = eroded

    cvuint8 = unsharp_mask(cvuint8)

    return cvuint8


def apply_hough(frame):
    global borders
    circles = hough_circles(frame)

    if not borders:
        hough_lines(frame)
        borders = True

    circles_arr = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            if (border_coordinates['ax'] - 3 < x < border_coordinates['bx'] + 3) and \
                    (border_coordinates['ay'] - 3 < y < border_coordinates['cy'] + 3):
                circles_arr.append(Circle(x, y, r))

    return circles_arr


def cut_circle(circle, frame):
    r = 10

    y1, y2 = max(0, circle.y - r), min(frame.shape[0], circle.y + r)
    x1, x2 = max(0, circle.x - r), min(frame.shape[1], circle.x + r)

    img = frame[y1:y2, x1:x2]

    if img.shape[0] != 2*r or img.shape[1] != 2*r:
        img_padded = np.full((2*r, 2*r, 3), 255, dtype=np.uint8)

        px1 = r - (circle.x - x1)
        px2 = px1 + (x2 - x1)
        py1 = r - (circle.y - y1)
        py2 = py1 + (y2 - y1)

        img_padded[py1:py2, px1:px2] = img
        img = img_padded

    mask = np.zeros((2*r, 2*r), dtype=np.uint8)
    cv.circle(mask, (r, r), r, 255, -1)

    final = np.full((2*r, 2*r, 3), 255, dtype=np.uint8)
    final[mask == 255] = img[mask == 255]

    return final


def get_circle_descriptors(final):
    mask = np.zeros((final.shape[0], final.shape[1]), dtype=np.uint8)
    cv.circle(mask, (final.shape[1]//2, final.shape[0]//2), 10, 255, -1)

    b_mean, g_mean, r_mean, _ = cv.mean(final, mask=mask)

    hsv = cv.cvtColor(final, cv.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean, _ = cv.mean(hsv, mask=mask)

    x_new = np.array([[b_mean, g_mean, r_mean, h_mean, s_mean, v_mean]])

    return x_new


def process_frame_model(frame, circle_index):
    global borders, border_coordinates
    original = frame.copy()

    cvuint8 = preprocess(original)
    circles_arr = apply_hough(cvuint8)

    for circle in circles_arr:
        final = cut_circle(circle, frame)
        x_new = get_circle_descriptors(final)

        with open('data/ball_descriptors.csv', mode='a') as balls_file:
            balls_writer = csv.writer(balls_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            balls_writer.writerow([f'c{circle_index}.png', x_new[0][0], x_new[0][1],x_new[0][2],x_new[0][3],x_new[0][4],x_new[0][5]])

        circle_index += 1

    return circle_index


def process_frame(frame, multi_model):
    global borders, border_coordinates
    original = frame.copy()

    cvuint8 = preprocess(original)
    circles_arr = apply_hough(cvuint8)

    cv.line(original, (border_coordinates['ax'], border_coordinates['ay']),
            (border_coordinates['bx'], border_coordinates['by']), (255, 0, 0), 3)
    cv.line(original, (border_coordinates['bx'], border_coordinates['by']),
            (border_coordinates['cx'], border_coordinates['cy']), (255, 0, 0), 3)
    cv.line(original, (border_coordinates['cx'], border_coordinates['cy']),
            (border_coordinates['dx'], border_coordinates['dy']), (255, 0, 0), 3)
    cv.line(original, (border_coordinates['ax'], border_coordinates['ay']),
            (border_coordinates['dx'], border_coordinates['dy']), (255, 0, 0), 3)

    for circle in circles_arr:
        final = cut_circle(circle, frame)
        x_new = get_circle_descriptors(final)

        data_pred = multi_model.predict(x_new)[0]

        if not ball_classes[data_pred] == 'trash':
            ball_color = np.flip(np.array(colors.to_rgb(ball_classes[data_pred])) * 255)
            cv.circle(original, (circle.x, circle.y), 10, ball_color, 2)
            cv.rectangle(original, (circle.x - 1, circle.y - 1), (circle.x + 1, circle.y + 1), (0, 128, 255),
                         -1)
            cv.putText(original, ball_classes[data_pred], (circle.x + 13, circle.y), cv.FONT_HERSHEY_PLAIN,
                       1, ball_color, 2, cv.LINE_4)

    return original


def start_capture(video_file):
    global frame_width, frame_height, size
    cap = cv.VideoCapture(video_file)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    return cap, size
