import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from line_detection import frame_line_extraction, frame_color_filter_and_hough
from color_filtering import color_filter
from blop_detection_lane import blob_detection_lane, draw_blobs
from perspective_modifier import perspective_modifier
from test_perspective_modification import situation_check

import matplotlib
matplotlib.use('TkAgg')


def hough_transform_video(video, lower, upper, threshold):
    cap = cv2.VideoCapture(video)

    # frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            mask = frame_line_extraction(frame, lower, upper, threshold)
            resize_frame = cv2.resize(frame, (500, 500))
            res = cv2.bitwise_and(resize_frame, resize_frame, mask=mask)
            cv2.imshow('frame', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def color_filtering_video(video, colors=['red', 'yellow']):
    cap = cv2.VideoCapture(video)

    # frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            filtered = color_filter(frame)
            cv2.imshow('frame', filtered)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

def video_color_filter_and_hough(video, threshold, frame_save=[], save_as_frames=False,
                                 save_as_video=False, save_path=None,
                                 frame_range=[]):
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
    all_hm = []
    pts = []

    situation = 'start'
    timer = 0
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):

        ret, frame = cap.read()
        # # if need to mirror the video:
        # frame = cv2.flip(frame,1)

        compt += 1
        if save_as_frames:
            frame_save = [compt]


        if ret == True:
            if compt in frame_range:
                # if compt < 205:
                #     situation = 'start'
                # if compt >= 205:
                #     situation = '5_15'
                # if compt >= 355:
                #     situation = '15_25'
                # if compt >= 500:
                #     situation = '25_35'
                # if compt >= 625:
                #     situation = '35_45'
                # if compt >= 680:
                #     situation = '45_50'


                filtered, lines = frame_color_filter_and_hough(frame, threshold)
                if len(lines) == 3 and np.max(lines[:, :, 1]) < 1.8:
                    all_lines.append(lines[np.argmin(lines[:, :, 0])][0])
                    detected, pts = blob_detection_lane(filtered, lines, situation=situation)


                if pts == []:
                    # print(compt)
                    from_above = np.zeros((1024, 1024, 3), np.uint8)
                    all_hm.append(np.zeros((3, 3)))
                else:
                    # blob_on_frame = draw_blobs(frame, pts)
                    # print(pts)
                    # cv2.imwrite("./media/test.jpg", blob_on_frame)
                    # from_above, hm, ordered_pts = perspective_modifier(pts[:4], blob_on_frame, (1024, 1024),
                    #                                                    situation=situation)
                    try:
                        blob_on_frame = draw_blobs(frame, pts)
                        from_above, hm, ordered_pts = perspective_modifier(pts[:4], blob_on_frame, (1024, 1024), situation=situation)
                        all_hm.append(hm)
                        current_situation = situation
                        if timer == 0:
                            situation = situation_check(frame, ordered_pts, current_situation)
                            if situation != current_situation:
                                timer = 40
                            if situation == '45_50' or situation == 'start':
                                timer += 30
                        # print(situation)
                    except:
                        from_above = np.ones((1024, 1024, 3), np.uint8)*255
                        all_hm.append(np.zeros((3, 3)))
                        print('it didnt work', compt)
                if timer != 0:
                    timer = timer - 1





                # displaying
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
    all_hm = np.array(all_hm)
    with open('all_homography_matrix.npy', 'wb') as f:
        np.save(f, all_hm)
    print(all_hm.shape)

    # all_lines = np.array(all_lines)
    # X = np.array([i for i in range(1, len(all_lines)+1)])
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    # axs[0].plot(X, all_lines[:, 0])
    # axs[1].plot(X,  all_lines[:, 1])
    # plt.show()
    # plt.plot(X, all_lines[:, 0])
    # plt.show()
    # plt.plot(X, all_lines[:, 1])
    # plt.show()

def extract_homography(video, threshold, frame_range=[]):
    cap = cv2.VideoCapture(video)

    frame_nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not frame_range:
        frame_range = [i for i in range(frame_nb)]
    compt = 0
    matrix_file_path = "./to_share/"
    M = np.zeros((3, 3))


    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        compt += 1
        if ret == True:
            if compt in frame_range:
                filtered, lines = frame_color_filter_and_hough(frame, threshold)
                detected, pts = blob_detection_lane(filtered, lines, situation='5_15')
                if pts == []:
                    print(compt)
                    from_above = np.zeros((1024, 1024, 3), np.uint8)
                else:
                    try:
                        from_above, M = perspective_modifier(pts[:4], frame, (1024, 1024), situation='5_15')
                    except:
                        from_above = np.zeros((1024, 1024, 3), np.uint8)
                # saving the homography matrix and the frame
                print(M)
                with open(matrix_file_path + 'matrix_'+ str(compt) + '.npy', 'wb') as f:
                    np.save(f, M)
                    np.save(f, frame)

                # cv2.imwrite(matrix_file_path + 'matrix_'+ str(compt) + ".jpg", from_above)

                # displaying
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame', 1200, 1200)
                cv2.imshow('frame', from_above)
                cv2.waitKey(-1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # video = "./videos/demi_50_brasse_titenis.mp4"
    video = "./videos/demi_50_brasse_titenis.mp4"
    save_path = "./videos/results/titenis_5-15_pers_modified.mp4"
    # for titenis
    lower = np.array([90, 130, 100])
    upper = np.array([105, 255, 255])
    threshold = 190

    # hough_transform_video(video, lower, upper, threshold)

    # color filtering
    # color_filtering_video(video)

    # Hough applied on color filtered image + clustering of lines
    # video_color_filter_and_hough(video, threshold, save_as_video=False, save_path=save_path)

    # 3 lanes test:
    video_color_filter_and_hough(video, threshold, frame_range=[i for i in range(0, 825)])

    # os.system('rm ./to_share/*')
    # extract_homography(video, threshold, frame_range=[i for i in range(208, 229)])
    #
    # with open('./to_share/matrix_2.npy', 'rb') as f:
    #     a = np.load(f)
    #     b = np.load(f)
    # print(a, b)


    # TODO tester l'algo sur https://www.dartfish.tv/Player?CR=p153270c361689m5580383&CL=1

    #TODO séprarer les lignes en réussissant à définir les zones
    # clustering sur le theta et de l'autre coté le bruit (en utilisant un dbscan)
    # peut-être après clustering des lignes (si ya du rouge garde la ligne on la garde sinon nan)

    # idée: garder le theta et ensuite faire un mask de zone carré sur toute la longeur
    # je le décale juste en translation
    # je fais ça pour toute l'image et ensuite je fusionne les masks
    # la distance serait un truc du genre l'écartement max qui arrive avec une diff de theta max
    #PAS BESOIN: je fais juste un mask de rouge avec l'image de base