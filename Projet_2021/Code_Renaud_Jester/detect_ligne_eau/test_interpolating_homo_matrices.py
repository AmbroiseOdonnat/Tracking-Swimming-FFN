import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def plot_all_hm(all_hm):
    fig, axs = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            axs[i, j].plot(all_hm[:, i, j], 'g*')
    plt.show()

def plot_all_hm_4_comp(all_hm, frame_range=None):
    if frame_range == None:
        frame_range = [0, len(all_hm)]
    i_max, j_max = (2, 2)
    fig, axs = plt.subplots(i_max, j_max)
    for i in range(i_max):
        for j in range(j_max):
            axs[i, j].plot(all_hm[frame_range[0]:frame_range[1], i, j], '*')
    plt.show()

def interpolate_hm(all_hm):
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
    X = np.resize(num_frame_non_zero, (len(num_frame_non_zero), 7))
    frames = np.resize(num_frame, (len(num_frame), 1))

    # cleaning of the data
    indexes = data_cleaning(all_hm_non_zero)
    clean_frame = num_frame_non_zero[indexes]
    cleared_hm = all_hm_non_zero[indexes]
    plot_all_hm(cleared_hm)

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
            z = np.polyfit(clean_frame, cleared_hm[:, i, j], 1)
            p = np.poly1d(z)
            inter = p(num_frame)
            if i == 1 and j == 2:
                inter = inter

            # do a different interpolation for the beginning
            beginning = 200
            beginning_frame = np.argwhere(indexes < beginning)
            z = np.polyfit(clean_frame[beginning_frame[:,0]], cleared_hm[beginning_frame[:,0], i, j], 1)
            p2 = np.poly1d(z)
            inter[:beginning] = p(num_frame[:beginning:])

            interpolated_hm[:, i, j] = inter
    return interpolated_hm

def video_hm_computed(video, hms, background=None):
    cap = cv2.VideoCapture(video)

    compt = 0



    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        # # if need to mirror the video:
        # frame = cv2.flip(frame,1)

        if ret == True and compt < len(hms):
            transformed_image = cv2.warpPerspective(frame, hms[compt], (1024, 1024))
            compt += 1
            if background != None:
                img_back = cv2.resize(cv2.imread(background), (1024, 1024))
                mask = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)
                img_back = cv2.bitwise_and(img_back, img_back, mask=mask)
                transformed_image = img_back + transformed_image


            # displaying
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1200, 1200)
            cv2.imshow('frame', transformed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


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

def apply_hm_inverse(video, img, hms):
    cap = cv2.VideoCapture(video)
    compt = 0

    # saving parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    file_name = './videos/test_apply_inv_homo.mp4'
    out = cv2.VideoWriter(file_name, fourcc, fps, (int(width), int(height)))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()

        # # if need to mirror the video:
        # frame = cv2.flip(frame,1)

        if ret == True and compt < len(hms):
            transformed_image = cv2.warpPerspective(img, np.linalg.inv(hms[compt]), (int(width), int(height))) #, cv2.WARP_INVERSE_MAP)
            compt += 1
            if compt > 200:
                out.write(transformed_image)

            # displaying
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 1200, 1200)
            cv2.imshow('frame', transformed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # matrices = 'all_homography_matrix.npy'
    # with open(matrices, 'rb') as f:
    #     all_hm = np.load(f)
    #     # print(all_hm.shape)
    #
    # non_zero_corner = []
    # for elmt in all_hm:
    #     if elmt[2, 2] != 0:
    #         non_zero_corner.append(True)
    #     else:
    #         non_zero_corner.append(False)
    # all_hm_non_zero = all_hm[non_zero_corner]
    # print(all_hm_non_zero.shape)
    # # plot_all_hm(all_hm_non_zero)
    # # plot_all_hm_4_comp(all_hm, [0, 799])
    # # plot_all_hm_4_comp(all_hm_non_zero)
    #
    # num_frame = np.arange(0, len(all_hm), 1)
    # num_frame_non_zero = num_frame[non_zero_corner]
    #
    #
    # # cleaning of the data
    # indexes = data_cleaning(all_hm_non_zero)
    # clean_frame = num_frame_non_zero[indexes]
    # cleared_hm = all_hm_non_zero[indexes]
    # # plot_all_hm(cleared_hm)
    # plt.plot(clean_frame, 'r*')
    # plt.show()
    #
    # # # interpolating using numpy
    # # z = np.polyfit(clean_frame, cleared_hm[:, 1, 0], 3)
    # # p = np.poly1d(z)
    # # inter_poly = p(num_frame)
    # # start_frame = 0
    # # plt.plot(num_frame[start_frame:], all_hm[start_frame:, 1, 0], 'r*')
    # # plt.plot(num_frame[start_frame:], inter_poly[start_frame:])
    # # plt.show()
    #
    #
    #
    #
    # # print(num_frame.shape)
    #
    # # for the (1, 1) coefficient of the matrix
    # X = np.resize(num_frame_non_zero, (len(num_frame_non_zero), 1))
    # frames = np.resize(num_frame, (len(num_frame), 1))
    #
    # # linear reg
    # # reg = LinearRegression().fit(X, all_hm_non_zero[:, 1, 1])
    # # results = reg.predict(frames)
    #
    # # ridge regression
    # clf = Ridge(alpha=1)
    # clf.fit(X, all_hm_non_zero[:, 1, 0])
    # results = clf.predict(frames)
    #
    # # # displaying
    # # plt.plot(num_frame[200:799], results[200:799])
    # # plt.plot(num_frame[200:799], all_hm[200:799, 1, 0], "r*")
    # # plt.show()
    #
    # colors = ['teal', 'yellowgreen', 'gold', 'red']
    # lw = 2
    # # for count, degree in enumerate([2, 3, 4, 5]):
    # #     model = make_pipeline(PolynomialFeatures(degree), Ridge())
    # #     model.fit(X[200:], all_hm_non_zero[200:, 1, 0])
    # #     y_plot = model.predict(frames)
    # #     plt.plot(frames, y_plot, color=colors[count], linewidth=lw,
    # #              label="degree %d" % degree)
    # # plt.plot(num_frame, all_hm[:, 1, 0], "r*")
    # # plt.show()
    #
    # # # interpolating using numpy
    # # z = np.polyfit(num_frame_non_zero[200:], all_hm_non_zero[200:, 1, 0], 3)
    # # p = np.poly1d(z)
    # # inter_poly = p(num_frame)
    # # start_frame = 200
    # # plt.plot(num_frame[start_frame:], all_hm[start_frame:, 1, 0], 'r*')
    # # plt.plot(num_frame[start_frame:], inter_poly[start_frame:])
    # # plt.show()
    #
    # # interpolating
    # interpolated = interpolate_hm(all_hm)
    # plot_all_hm(interpolated)
    # # #
    # video = "./videos/demi_50_brasse_titenis.mp4"
    # background = "./media/Swimming_pool_50m_above.png"
    # video_hm_computed(video, interpolated, background=background)
    # # apply_hm_inverse(video, cv2.resize(cv2.imread(background), (1024, 1024)), interpolated)


    video = "./videos/gwandju_returned.mp4"
    background = "./media/Swimming_pool_50m_above.png"
    matrices = './videos/results/gwandju_returned_hm.npy'
    with open(matrices, 'rb') as f:
        all_hm = np.load(f)

    video_hm_computed(video, all_hm, background=background)
