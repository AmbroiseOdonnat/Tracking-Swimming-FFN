import numpy as np
import cv2
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from line_detection_frame import frame_color_filter_and_hough
from blop_detection_lane import select_outside_lines


def interpolate_hm(all_hm, poly_deg=7):
    non_zero_corner = []
    for elmt in all_hm:
        if elmt[2, 2] != 0:
            non_zero_corner.append(True)
        else:
            non_zero_corner.append(False)
    all_hm_non_zero = all_hm[non_zero_corner]
    num_frame = np.arange(0, len(all_hm), 1)
    num_frame_non_zero = num_frame[non_zero_corner]
    # formatting for the fitting and predicting
    X = np.resize(num_frame_non_zero, (len(num_frame_non_zero), 1))
    frames = np.resize(num_frame, (len(num_frame), 1))

    # cleaning of the data
    indexes = data_cleaning(all_hm_non_zero)
    clean_frame = num_frame_non_zero[indexes]
    cleared_hm = all_hm_non_zero[indexes]

    interpolated_hm = np.ones(all_hm.shape)
    for i in range(3):
        for j in range(3):
            if i == 3 and j == 3:
                break

            # BEWARE: the 200 is because the first frames seems too be poorly detected (see graphs)
            # TODO correct the five meters detection technique
            # model = Ridge(alpha=1)
            # model.fit(X[200:], all_hm_non_zero[200:, i, j])

            # model = make_pipeline(PolynomialFeatures(2), Ridge())
            # model.fit(X[:], all_hm_non_zero[:, 1, 0])
            #
            # inter = model.predict(frames)

            # with numpy
            z = np.polyfit(clean_frame, cleared_hm[:, i, j], poly_deg)
            p = np.poly1d(z)
            inter = p(num_frame)

            interpolated_hm[:, i, j] = inter
    return interpolated_hm


def reject_outliers_2(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return np.argwhere(s < m)


def data_cleaning(all_non_zero_hm):
    indexes = np.argwhere(all_non_zero_hm[:, 2, 2] > 0)
    for i in range(3):
        for j in range(3):
            if i + j != 4:
                indexes = np.intersect1d(indexes, reject_outliers_2(all_non_zero_hm[:, i, j]))
    return indexes

def post_process_vertical_correction(img, hm, lines_original,threshold=100):
    try:
        _, lines = frame_color_filter_and_hough(img, threshold)
    except IndexError:
        print("no puedo")
        return hm

    lane_side = select_outside_lines(lines)
    line_6 = lane_side[np.argmax(lane_side[:,:,0])]
    rho, theta = line_6[0]
    # x_or, y_or, _ = np.dot(np.linalg.inv(hm), np.resize(np.array([0, rho, 1], dtype=float), (3,1)))
    # x_or, y_or = x_or[0], y_or[0]
    #
    # s = x_or*hm[2, 0] + y_or*hm[2, 1] + 1
    # hm[1, 2] += hm[1, 2] + rho - s*img.shape[0]*3/8

    T = np.eye(3)
    T[1, 2] = rho - img.shape[0]*3/8
    print(rho - img.shape[0]*3/8)

    return T @ hm

# code to use some post processing:

# def saving_corrected(video, threshold, save_path, frame_stop=100000, timer_cap=40):
#     """Input: video, threshold for the hough transform, frame to stop, the path where to save video
#     Output: save the video after filtering
#
#     this function saves the video of all the filtered image using the filter color and the hough transform.
#
#     To use for demo and testing purposes.
#     """
#     cap = cv2.VideoCapture(video)
#     compt = 0
#     timer = 0
#     situation = 'start'
#     all_hm = []
#     all_lines = []
#
#     # saving parameters
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     file_name = video.split('/')[-1].split('.')[0] + '_inter_and_correct.mp4'
#     out = cv2.VideoWriter(save_path + file_name, fourcc, fps,(int(1024), int(1024)))
#
#     # Check if camera opened successfully
#     if cap.isOpened() == False:
#         print("Error opening video stream or file")
#
#     while (cap.isOpened()) and compt < frame_stop:
#         ret, frame = cap.read()
#
#         compt += 1
#         if timer != 0:
#             timer = timer - 1
#         if ret == True:
#             try:
#                 filtered, lines = frame_color_filter_and_hough(frame, threshold)
#                 all_lines.append(lines)
#             except IndexError: # which means we detect no lines
#                 print('No lines found', compt)
#                 break
#
#             hm = np.zeros((3, 3))
#             # blob detection and use of the detection to evaluate the hommography matrix
#             # we have the assumption that the lines are 3 and that they shouldn't have theta too high (more and pi/2)
#             if well_detected_lines(lines):
#                 # detection of the blob
#                 detected, pts = blob_detection_lane(filtered, lines, situation=situation)
#
#                 # computing the homography matrix if the pts seems ok
#                 if len(pts) == 4:
#                     try:
#                         from_above, hm, ordered_pts = perspective_modifier(pts, frame, (1024, 1024), situation=situation)
#
#                         # update the situation when needed and use of a timer not to change situation too many times
#                         current_situation = situation
#                         if timer == 0:
#                             situation = situation_check(frame, ordered_pts, current_situation)
#                             # print(situation, compt)
#                             if situation != current_situation:
#                                 timer = timer_cap
#                                 if situation == '45_50' or situation == 'start':
#                                     timer += 10
#
#                     except IndexError:
#                         pass
#             all_hm.append(hm)
#
#         else:
#             break
#     cap.release()
#
#     print("Extraction of the homography matrices using Hough lines: Done")
#
#     all_hm = interpolate_hm(np.array(all_hm))
#     all_lines = np.array(all_lines)
#
#     print("Interpolation: Done")
#
#     # see the result on the original video
#     compt = 0
#     cap = cv2.VideoCapture(video)
#     while (cap.isOpened()):
#         compt += 1
#         ret, frame = cap.read()
#
#         if ret == True and compt < len(all_hm):
#             transformed_image = cv2.warpPerspective(frame, all_hm[compt], (1024, 1024))
#             if len(all_lines[compt]) == 3:
#                 hm = post_process_vertical_correction(transformed_image, all_hm[compt], all_lines[compt])
#
#                 transformed_image = cv2.warpPerspective(frame, hm, (1024, 1024))
#
#             # insert the background
#             img_back = cv2.resize(cv2.imread("./media/Swimming_pool_50m_above.png"), (1024, 1024))
#             mask = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
#             _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
#             img_back = cv2.bitwise_and(img_back, img_back, mask=mask)
#             transformed_image = img_back + transformed_image
#
#             out.write(transformed_image)
#         else:
#             break
#     cap.release()
#     print("Resulting video stored at: " + save_path + file_name)