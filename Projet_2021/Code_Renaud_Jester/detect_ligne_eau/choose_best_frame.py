import numpy as np
import cv2


from line_detection_frame import frame_color_filter_and_hough, well_detected_lines
from blop_detection_lane import blob_detection_lane
from perspective_modifier import perspective_modifier, situation_check
from interpolation_homomatrices import interpolate_hm


def read_video(video):
    cap = cv2.VideoCapture(video)

    compt = 0
    to_save = []

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        compt += 1
        if ret == True:
            # displaying
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1200, 1200)
            cv2.imshow('frame', frame)
            # save = input('something')
            if cv2.waitKey(0) == ord('y'):
                to_save.append(compt)
            else:
                continue

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return to_save

if __name__ == '__main__':
    video = "./videos/results/50_brasse_stevens_from_above.mp4"
    to_save = read_video(video)
    print(to_save)