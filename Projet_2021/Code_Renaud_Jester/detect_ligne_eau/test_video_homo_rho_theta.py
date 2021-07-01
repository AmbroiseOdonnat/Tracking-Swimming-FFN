import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from line_detection import frame_line_extraction, frame_color_filter_and_hough
from color_filtering import color_filter
from blop_detection_lane import blob_detection_lane, select_outside_lines
from perspective_modifier import perspective_modifier, corners_ordering
from homography_matrix_update import point_update
from blop_detection_lane import draw_blobs


import matplotlib
matplotlib.use('TkAgg')


def video_color_filter_and_hough(video, threshold, frame_save=[],
                                 save_as_frames=False, save_as_video=False,
                                 save_path=None, frame_range=[]):
    cap = cv2.VideoCapture(video)

    # frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if save_as_video:
        # out = out = cv2.VideoWriter(save_path,fourcc, fps, (int(height) ,int(width)))
        # out = cv2.VideoWriter(save_path,fourcc, fps, (int(width) ,int(height)))
        # out for perspective modified images:
        out = cv2.VideoWriter(save_path, fourcc, fps, (1024, 1024))
    compt = 0
    all_lines = []
    all_pts = []


    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        compt += 1
        if save_as_frames:
            frame_save = [compt]


        if ret == True:
            if compt in frame_range:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
                if len(lines) == 3 and np.max(lines[:, :, 1]) < 1.8:
                    all_lines.append(lines[np.argmin(lines[:, :, 0])][0])
                    detected, pts = blob_detection_lane(filtered, lines)

                if pts == []:
                    print(compt)
                    from_above = np.zeros((1024, 1024, 3), np.uint8)
                else:
                    # get all the points to print them later
                    pts = np.array(pts)
                    pts = np.resize(pts, (pts.shape[0], pts.shape[-1]))
                    # order the corners
                    pts = corners_ordering(pts)
                    all_pts.append(pts[1])
                    try:
                        from_above, hm = perspective_modifier(pts[:4], frame, (1024, 1024))
                        print(lines)
                        print(pts)
                        print(hm)
                    except:
                        from_above = np.zeros((1024, 1024, 3), np.uint8)


                #displaying
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame', 1200, 1200)
                # cv2.imshow('frame', color_filter(filtered, colors=['redred']))
                cv2.imshow('frame', from_above)

                # cv2.waitKey(-1)
                if compt in frame_save:
                    cv2.imwrite("./media/3lanes_"+str(compt)+".jpg", filtered)
                    cv2.imwrite("./media/3lanes_"+str(compt)+"_entireimg.jpg", frame)
                    print(lines)
                if save_as_video:
                    out.write(from_above)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    all_lines = np.array(all_lines)
    X = np.array([i for i in range(1, len(all_lines)+1)])
    fig, axs = plt.subplots(5)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(X, all_lines[:, 0])
    axs[1].plot(X,  all_lines[:, 1])

    all_pts = np.array(all_pts)
    print(all_pts)
    xs, ys = multiple_updates_one_line(893, 210, all_lines)
    axs[2].plot(X, xs, '-r')
    axs[2].plot(X, all_pts[:, 0])

    axs[3].plot(X, ys, '-r')
    axs[3].plot(X, all_pts[:, 1])

    axs[4].plot(xs, ys, '*r')
    axs[4].plot(all_pts[:, 0], all_pts[:, 1], '*')


    plt.show()
    # plt.plot(X, all_lines[:, 0])
    # plt.show()
    # plt.plot(X, all_lines[:, 1])
    # plt.show()



def video_hough_hm_update(video, threshold, frame_save=[],
                                 save_as_frames=False, save_as_video=False,
                                 save_path=None, frame_range=[], pts_init=np.array([]),
                          lines_init=np.array([]), hm_init = np.array([])):
    cap = cv2.VideoCapture(video)

    # frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    if save_as_video:
        # out = out = cv2.VideoWriter(save_path,fourcc, fps, (int(height) ,int(width)))
        # out = cv2.VideoWriter(save_path,fourcc, fps, (int(width) ,int(height)))
        # out for perspective modified images:
        out = cv2.VideoWriter(save_path, fourcc, fps, (1024, 1024))
    compt = 0
    all_lines = []
    all_pts = []


    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # initialisation for the frame 80
    pts = pts_init
    prev_lines = lines_init
    hm = hm_init
    all_pts = [pts_init]

    while (cap.isOpened()):
        ret, frame = cap.read()



        compt += 1
        if save_as_frames:
            frame_save = [compt]


        if ret == True:
            if compt in frame_range:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
                if len(lines) == 3 and np.max(lines[:, :, 1]) < 1.8:
                    new_lines = select_outside_lines(lines)
                    # print(new_lines)
                    new_pts = update_all_points(pts_init, lines_init, new_lines)
                    all_pts.append(new_pts)
                    # print(pts)
                    # print(new_pts)
                    pts = new_pts
                    prev_lines = new_lines
                    blob_on_frame = draw_blobs(frame, np.expand_dims(pts, 1).astype( int))
                    # pts.resize((pts.shape[0], 1, pts.shape[-1]
                    try:
                        from_above, hm, _ = perspective_modifier(pts, blob_on_frame, (1024, 1024), situation='5_15')
                    except IndexError as a:
                        print(pts)
                        print(new_lines)
                        print(a)
                        break

                else:
                    from_above = cv2.warpPerspective(frame, hm, (1024, 1024))

                # displaying
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame', 1200, 1200)
                # cv2.imshow('frame', color_filter(filtered, colors=['redred']))
                cv2.imshow('frame', blob_on_frame)


                # saving
                if compt in frame_save:
                    cv2.imwrite("./media/3lanes_"+str(compt)+".jpg", filtered)
                    cv2.imwrite("./media/3lanes_"+str(compt)+"_entireimg.jpg", frame)
                    print(lines)
                if save_as_video:
                    out.write(from_above)

                # the enf
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # cv2.waitKey(-1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    all_pts = np.array(all_pts)
    X = np.array([i for i in range(1, len(all_pts)+1)])
    fig, axs = plt.subplots(4)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(X, all_pts[:, 0, 0])
    axs[1].plot(X,  all_pts[:, 1, 0])
    axs[2].plot(X, all_pts[:, 2, 0])
    axs[3].plot(X, all_pts[:, 3, 0])
    plt.show()



def multiple_updates_one_line(x_init, y_init, lines):
    xs = [x_init]
    ys = [y_init]
    x2, y2 = x_init, y_init
    for i in range(len(lines)-1):
        x1, y1 = x2, y2
        rho1, theta1 = lines[i]
        rho2, theta2 = lines[i + 1]
        x2, y2 = point_update(x1, y1, rho1, theta1, rho2, theta2)
        xs.append(x2)
        ys.append(y2)
    return xs, ys

def update_all_points(pts_ordered, previous_lines, new_lines):
    """expected that the lines and the points are ordered as stated in order_corners and select_outside_lanes"""
    new_points = []
    new_points.append(point_update(pts_ordered[0, 0], pts_ordered[0, 1], previous_lines[0,0,0],
                                   previous_lines[0,0,1], new_lines[0,0,0], new_lines[0,0,1]))

    new_points.append(point_update(pts_ordered[1, 0], pts_ordered[1, 1], previous_lines[0, 0, 0],
                                   previous_lines[0,0,1], new_lines[0,0,0], new_lines[0,0,1]))

    new_points.append(point_update(pts_ordered[2, 0], pts_ordered[2, 1], previous_lines[1,0,0],
                                   previous_lines[1,0,1], new_lines[1,0,0], new_lines[1,0,1]))

    new_points.append(point_update(pts_ordered[3, 0], pts_ordered[3, 1], previous_lines[1,0,0],
                                   previous_lines[1,0,1], new_lines[1,0,0], new_lines[1,0,1]))
    return np.array(new_points)



if __name__ == "__main__":

    video = "./videos/demi_50_brasse_titenis.mp4"
    save_path = "./videos/results/titenis_5-15_pers_modified.mp4"
    # for titenis
    threshold = 190

    # video_color_filter_and_hough(video, threshold, frame_range=[i for i in range(225, 226)])

    # to begin frame 80 on the short video
    pts = np.array([[108,  310], [893,  210], [1085, 322], [216,  444]])
    lines = np.array([[[322.7143, 1.4461299]], [[469.75, 1.4333516]]])
    hm = np.array([[ 3.33414963e-01,  6.67670838e-01,  7.43006842e+02],
                  [ 3.58426249e-01,  3.17509939e+00, -4.94779979e+02],
                  [ 4.31874440e-05,  1.30289699e-03,  1.00000000e+00]])

    # to begin frame 225 on the big video:
    lines = np.array([[[324.42856,     1.4423898]], [[470.125,      1.4344425]]])
    pts = np.array([[113,  309], [908,  208], [1092, 323], [220,  445]])
    hm = np.array([[3.15383981e-01,  5.74419818e-01,  7.54425973e+02],
                   [3.38517137e-01,  3.01260992e+00, - 4.50233994e+02],
                   [2.80029856e-05, 1.12678500e-03, 1.00000000e+00]])

    video_hough_hm_update(video, threshold, frame_range=[i for i in range(225, 450)],
                          pts_init=pts, lines_init=lines, hm_init=hm)

    # TODO use the same but not updating the x1, y1, rho1, theta1
    # for no propagation of the error
