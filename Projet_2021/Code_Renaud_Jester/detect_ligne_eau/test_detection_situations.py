import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from line_detection import frame_color_filter_and_hough

def video_extract_lines(video, threshold, frame_save=[],
                                 save_as_frames=False, save_as_video=False,
                                 save_path=None, frame_range=[]):
    cap = cv2.VideoCapture(video)

    compt = 0
    all_lines = []


    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        compt += 1


        if ret == True:
            if compt in frame_range:
                try:
                    filtered, lines = frame_color_filter_and_hough(frame, threshold)
                except IndexError:
                    print('End of the race')
                    break
                if len(lines) == 3 and np.max(lines[:, :, 1]) < 1.8:
                    all_lines.append(lines[np.argmin(lines[:, :, 0])][0])
                else:
                    all_lines.append([0, 0])

                # displaying
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame', 1200, 1200)
                # cv2.imshow('frame', color_filter(filtered, colors=['redred']))
                cv2.imshow('frame', filtered)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


    all_lines = np.array(all_lines)
    with open('all_lines.npy', 'wb') as f:
        np.save(f, all_lines)
    print(all_lines.shape)

    # plt.plot(X, all_lines[:, 0])
    # plt.show()
    # plt.plot(X, all_lines[:, 1])
    # plt.show()

if __name__ == "__main__":
    video = "./videos/gwanju.mp4"
    threshold = 190

    top = time.time()
    video_extract_lines(video, threshold, frame_range=[i for i in range(16000)])
    print(time.time() - top)

    with open('all_lines.npy', 'rb') as f:
        all_lines = np.load(f)


    X = np.array([i for i in range(1, len(all_lines)+1)])
    non_zero_lines = []
    for elmt in all_lines:
        if elmt[1] != 0:
            non_zero_lines.append(True)
        else:
            non_zero_lines.append(False)

    all_lines_non_zero = all_lines[non_zero_lines]
    mask_index = [True if (i%50 == 0) else False for i in range(len(all_lines_non_zero)) ]
    half_lines = all_lines_non_zero[mask_index]
    half_frames = X[non_zero_lines][mask_index]

    # # displaying
    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(X, all_lines[:, 0], '*')
    axs[1].plot(X,  all_lines[:, 1], '*')
    axs[2].plot(np.diff(half_lines[:, 1]))
    plt.show()

    # diff = np.diff(half_lines[:, 1])
    # z = np.polyfit(half_frames[1:], diff, 4)
    # p = np.poly1d(z)
    # inter = p(X)
    # plt.plot(X, inter)
    # plt.plot(half_frames[1:], diff, 'r+')
    # plt.show()

    non_zero_frames = X[non_zero_lines]
    z = np.polyfit(non_zero_frames, all_lines_non_zero[:, 1], 4)
    p = np.poly1d(z)
    inter = p(X)
    plt.plot(X, inter)
    plt.plot(non_zero_frames, all_lines_non_zero[:, 1], 'r+')
    plt.show()

