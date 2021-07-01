import numpy as np
import cv2
import matplotlib.pyplot as plt

import pysift
import time



def close_to_the_side(img, pts, side='right'):
    if side == 'right':
        first_point = pts[1]
        second_point = pts[2]
        border = img.shape[1]
    else:
        first_point = pts[0]
        second_point = pts[3]
        border = 0
    diff_first = abs(border - first_point[0])
    diff_second = abs(border - second_point[0])
    return diff_first <= 12 or diff_second <= 12

def situation_check(img, pts, current_situation):
    situations_transition = {
        'start':'5_15',
        '5_15':'15_25',
        '15_25':'25_35',
        '25_35':'35_45',
        '35_45':'45_50',
        '45_50':'45_35',
        '45_35': '35_25',
        '35_25':'25_15',
        '25_15':'15_5',
        '15_5':'start'
    }

    if current_situation in ['start', '5_15', '15_25', '25_35', '35_45']:
        close = close_to_the_side(img, pts, side='right')
    else:
        close = close_to_the_side(img, pts, side='left')

    if close:
        return situations_transition[current_situation]
    else:
        return current_situation


if __name__ == "__main__":
    path = "./media/"
    file_name = "brasse_50_fframe.png"

    img = cv2.imread(path + file_name)
    rows, cols, ch = img.shape
    size_out = 2000

    # # points of interest on the corners
    # pts1 = np.float32([[2206, 1096], [3745, 1750], [959, 1312], [2184, 2096]])
    # pts1 = np.float32([[2206, 1096], [3745, 1750], [36, 1538], [710, 2150]])
    # pts2 = np.float32([[size_out, 0], [size_out, size_out], [0, 0], [0, size_out]])

    # # points of interest on the middle line
    # number_lanes = 10
    # half_pool = size_out//2
    # fourth_line = 4*size_out//number_lanes
    # fifteen_mark = 10*size_out//25
    # pts1 = np.float32([[294, 1769], [1365, 1572], [390, 1861], [1483, 1651]])
    # pts2 = np.float32([[0, fourth_line], [fifteen_mark, fourth_line], [0, half_pool], [fifteen_mark, half_pool]])

    # points of interest from the 15m
    pts1 = np.float32([[955, 1308], [3733, 1757], [2201, 1104], [2184, 2104]])
    pts2 = np.float32([[size_out // 2, 0], [size_out, size_out], [size_out, 0], [size_out // 2, size_out]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (size_out, size_out))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    # plt.imshow(img)
    plt.show(block=True)

    cv2.imwrite('./media/halfspa.jpg', dst)

    # # sift test
    # comp_time = time.time()
    # img = cv2.imread(path + file_name)
    # resized = np.ones((300,300,3),np.uint8)*255
    # img = cv2.resize(img, (300, 300))
    # plt.imshow(img)
    # plt.show(block=True)
    # gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # sift = cv2.xfeatures2d.SIFT_create()
    # # kp = sift.detect(gray, None)
    #
    # keypoints, descriptors = pysift.computeKeypointsAndDescriptors(gray)
    #
    # img = cv2.drawKeypoints(gray, keypoints, img)
    # plt.imshow(img)
    # plt.show(block=True)
    #
    # print(time.time() - comp_time)
