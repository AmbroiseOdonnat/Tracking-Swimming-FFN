import numpy as np
import cv2
import argparse

from line_detection_frame import frame_color_filter_and_hough, well_detected_lines
from blop_detection_lane import blob_detection_lane
from perspective_modifier import perspective_modifier, situation_check
from interpolation_homomatrices import interpolate_hm, post_process_vertical_correction

def hm_exctractor(video, threshold, save_path, frame_stop=100000, timer_cap=40):
    """Input: video, threshold for the hough transform, frame to stop, the path where to save the matrices
    Output: none in python but a .npy file is written at the save_path with all the homography matrices
    for the frame 0 to frame_stop.

    This function is the funnction to use in order to only get the homography matrices of the video in order to use
    them for tracking or visualization.

    For the moment the matrices are stored as a numpy file that can be opened using:
    with open('../my_video_hm.npy', 'rb') as f:
        all_hm = np.load(f)
    """
    cap = cv2.VideoCapture(video)
    compt = 0
    timer = 0
    situation = 'start'
    all_hm = []

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()) and compt < frame_stop:
        ret, frame = cap.read()

        compt += 1
        if timer != 0:
            timer = timer - 1

        if ret == True:
            try:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
            except IndexError: # which means we detect no lines
                print('No lines found', compt)
                break

            hm = np.zeros((3, 3))

            # blob detection and use of the detection to evaluate the hommography matrix
            # we have the assumption that the lines are 3 and that they shouldn't have theta too high (more and pi/2)
            if well_detected_lines(lines):
                # detection of the blob
                try:
                    detected, pts = blob_detection_lane(filtered, lines, situation=situation)
                except ValueError:
                    print('Considered the end of the video')
                    break

                # computing the homography matrix if the pts seems ok
                if len(pts) == 4:
                    try:
                        _, hm, ordered_pts = perspective_modifier(pts, frame, (1024, 1024), situation=situation)

                        # update the situation when needed and use of a timer not to change situation too many times
                        current_situation = situation
                        if timer == 0:
                            situation = situation_check(frame, ordered_pts, current_situation)
                            if situation != current_situation:
                                timer = timer_cap
                                if situation == '45_50' or situation == 'start' :
                                    timer += 10
                    except IndexError:
                        pass
            # hm is zero if: the lines are not well detected or the blob are not well positioned or clustered
            # it is mostly the second option
            all_hm.append(hm)
        else:
            break
    cap.release()
    print("Extraction of the homography matrices using Hough lines: Done")

    all_hm = interpolate_hm(np.array(all_hm))

    print("Interpolation: Done")
    with open(save_path + video.split('/')[-1].split('.')[0] + '_hm.npy', 'wb') as f:
        np.save(f, all_hm)
    print("Homography matrices stored in: " + save_path + video.split('/')[-1].split('.')[0] + '_hm.npy')

def saving_filtered(video, threshold, save_path, frame_stop=100000):
    """Input: video, threshold for the hough transform, frame to stop, the path where to save video
    Output: save the video after filtering

    this function saves the video of all the filtered image using the filter color and the hough transform.

    To use for demo and testing purposes.
    """
    cap = cv2.VideoCapture(video)
    compt = 0

    # saving parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    file_name = video.split('/')[-1].split('.')[0] + '_filtered.mp4'
    out = cv2.VideoWriter(save_path + file_name, fourcc, fps, (int(width) ,int(height)))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()) and compt < frame_stop:
        ret, frame = cap.read()

        compt += 1
        if ret == True:
            try:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
            except IndexError: # which means we detect no lines
                print('No lines found', compt)
                break
            out.write(filtered)
        else:
            break

    cap.release()

def saving_from_above(video, threshold, save_path, frame_stop=100000, timer_cap=40):
    """Input: video, threshold for the hough transform, frame to stop, the path where to save video
    Output: save the video after filtering

    this function saves the video of all the filtered image using the filter color and the hough transform.

    To use for demo and testing purposes.
    """
    cap = cv2.VideoCapture(video)
    compt = 0
    timer = 0
    situation = 'start'

    # saving parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    file_name = video.split('/')[-1].split('.')[0] + '_from_above.mp4'
    out = cv2.VideoWriter(save_path + file_name, fourcc, fps,(int(1024), int(1024)))
    file_name2 = video.split('/')[-1].split('.')[0] + '_blob_detected.mp4'
    out2 = cv2.VideoWriter(save_path + file_name2, fourcc, fps, (int(width) ,int(height)))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()) and compt < frame_stop:
        ret, frame = cap.read()

        compt += 1
        if timer != 0:
            timer = timer - 1
        if ret == True:
            try:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
            except IndexError: # which means we detect no lines
                print('No lines found', compt)
                break

            from_above = np.zeros((1024, 1024, 3), np.uint8)
            detected = np.zeros(frame.shape, np.uint8)
            # blob detection and use of the detection to evaluate the hommography matrix
            # we have the assumption that the lines are 3 and that they shouldn't have theta too high (more and pi/2)
            if well_detected_lines(lines):
                # detection of the blob
                try:
                    detected, pts = blob_detection_lane(filtered, lines, situation=situation)
                except ValueError:
                    print('Considered the end of the video')
                    break
                # computing the homography matrix if the pts seems ok
                if len(pts) == 4:
                    try:
                        from_above, _, ordered_pts = perspective_modifier(pts, frame, (1024, 1024), situation=situation)

                        # update the situation when needed and use of a timer not to change situation too many times
                        current_situation = situation
                        if timer == 0:
                            situation = situation_check(frame, ordered_pts, current_situation)
                            # print(situation, compt)
                            if situation != current_situation:
                                timer = timer_cap
                                if situation == '45_50' or situation == 'start' :
                                    timer += 10
                    except IndexError:
                        pass
            out.write(from_above)
            out2.write(detected)

        else:
            break

    cap.release()

def saving_interpolated(video, threshold, save_path, frame_stop=100000, timer_cap=40):
    """Input: video, threshold for the hough transform, frame to stop, the path where to save video
    Output: save the video after filtering

    this function saves the video of all the filtered image using the filter color and the hough transform.

    To use for demo and testing purposes.
    """
    cap = cv2.VideoCapture(video)
    compt = 0
    timer = 0
    situation = 'start'
    all_hm = []

    # saving parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    file_name = video.split('/')[-1].split('.')[0] + '_interpolated.mp4'
    out = cv2.VideoWriter(save_path + file_name, fourcc, fps,(int(1024), int(1024)))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()) and compt < frame_stop:
        ret, frame = cap.read()

        compt += 1
        if timer != 0:
            timer = timer - 1
        if ret == True:
            try:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
            except IndexError: # which means we detect no lines
                print('No lines found', compt)
                break

            hm = np.zeros((3, 3))
            # blob detection and use of the detection to evaluate the hommography matrix
            # we have the assumption that the lines are 3 and that they shouldn't have theta too high (more and pi/2)
            if well_detected_lines(lines):
                # detection of the blob
                try:
                    detected, pts = blob_detection_lane(filtered, lines, situation=situation)
                except ValueError:
                    print('Considered the end of the video')
                    break

                # computing the homography matrix if the pts seems ok
                if len(pts) == 4:
                    try:
                        from_above, hm, ordered_pts = perspective_modifier(pts, frame, (1024, 1024), situation=situation)

                        # update the situation when needed and use of a timer not to change situation too many times
                        current_situation = situation
                        if timer == 0:
                            situation = situation_check(frame, ordered_pts, current_situation)
                            # print(situation, compt)
                            if situation != current_situation:
                                timer = timer_cap
                                if situation == '45_50' or situation == 'start':
                                    timer += 10
                    except IndexError:
                        pass
            all_hm.append(hm)

        else:
            break
    cap.release()

    print("Extraction of the homography matrices using Hough lines: Done")

    all_hm = interpolate_hm(np.array(all_hm))

    print("Interpolation: Done")

    # see the result on the original video
    compt = 0
    cap = cv2.VideoCapture(video)
    while (cap.isOpened()):
        compt += 1
        ret, frame = cap.read()

        if ret == True and compt < len(all_hm):
            transformed_image = cv2.warpPerspective(frame, all_hm[compt], (1024, 1024))

            # insert the background
            img_back = cv2.resize(cv2.imread("./media/Swimming_pool_50m_above.png"), (1024, 1024))
            mask = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
            img_back = cv2.bitwise_and(img_back, img_back, mask=mask)
            transformed_image = img_back + transformed_image

            out.write(transformed_image)
        else:
            break
    cap.release()
    print("Resulting video stored at: " + save_path + file_name)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for homography matrix computation algorithm.')
    parser.add_argument('--video', help='Path to the video.')
    parser.add_argument('--usage', help='filter or extract', default='extract')
    parser.add_argument('--save', help='Path where to save the result.')
    parser.add_argument('--threshold', help='threshold for hough transform', default='190')
    parser.add_argument('--stop', help='Number of the frame to stop the algo.', default='100000')
    args = parser.parse_args()

    # example of command: python swimming_video_homography.py --video ./videos/demi_50_brasse_titenis.mp4
    # --save ./videos/results/ --usage filter --stop 800
    if args.usage == 'filter':
        saving_filtered(args.video, int(args.threshold), args.save, frame_stop= int(args.stop))
    elif args.usage == 'above':
        saving_from_above(args.video, int(args.threshold), args.save, frame_stop= int(args.stop))
    # saving_from_above("./videos/demi_50_brasse_titenis.mp4", 190, 800, "./videos/results/")
    elif args.usage == 'inter':
        saving_interpolated(args.video, int(args.threshold), args.save, frame_stop= int(args.stop))
    # elif args.usage == 'correct':
    #     saving_corrected(args.video, int(args.threshold), args.save, frame_stop= int(args.stop))
    elif args.usage == 'extract':
        hm_exctractor(args.video, int(args.threshold), args.save)