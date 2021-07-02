import os
import csv
from billard import ball_classifier, circle_ml
import cv2 as cv

render_output = True
build_model = True

def main():
    if build_model:
        print(os.getcwd())

        with open('data/ball_descriptors.csv', mode='w') as balls_file:
            balls_writer = csv.writer(balls_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            balls_writer.writerow(['circle_index', 'b_int', 'g_int', 'r_int', 'h_int', 's_int', 'v_int'])

        cap, frame_size = ball_classifier.start_capture('DIP/video.mp4')
        circle_index = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                circle_index = ball_classifier.process_frame_model(frame, circle_index)

            else:
                break
        circle_ml.build_model('data/ball_descriptors.csv')

    cap, frame_size = ball_classifier.start_capture('DIP/video.mp4')
    model = ball_classifier.load_model('model/balls_model.pkl')

    result = None
    if render_output:
        result = cv.VideoWriter('DIP/output.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                24, frame_size)
    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            output = ball_classifier.process_frame(frame, model)

            if render_output:
                result.write(output)

            cv.imshow('Frame', output)

            if cv.waitKey(1) & 0xFF == ord('s'):
                break

        else:
            break

    cap.release()

    if render_output:
        result.release()
    cv.destroyAllWindows()


main()
