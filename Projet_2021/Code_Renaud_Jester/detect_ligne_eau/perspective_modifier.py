import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from line_detection import print_image

def swimming_pool_dic(shape_output_img, type=50):
    """Dictionary containing the coordinates (value) of the key points on a lane (keys)
    the key indicates also the number of the line
    415 is the coordinates of the 4th lane on the 15m mark
    I asssume that there are always 8 swimmers not more (even if there exits swimming pools with 10 lanes"""
    h, w = shape_output_img
    dic = {
        400: [          0, 5 * h // 8],
        405: [    w // 10, 5 * h // 8],
        415: [3 * w // 10, 5 * h // 8],
        425: [    w // 2 , 5 * h // 8],
        435: [7 * w // 10, 5 * h // 8],
        445: [9 * w // 10, 5 * h // 8],
        450: [w          , 5 * h // 8],
        500: [0, 4 * h // 8],
        505: [w // 10, 4 * h // 8],
        515: [3 * w // 10, 4 * h // 8],
        525: [w // 2, 4 * h // 8],
        535: [7 * w // 10, 4 * h // 8],
        545: [9 * w // 10, 4 * h // 8],
        550: [w, 4 * h // 8],
        600: [         0 , 3 * h // 8],
        605: [    w // 10, 3 * h // 8],
        615: [3 * w // 10, 3 * h // 8],
        625: [    w // 2 , 3 * h // 8],
        635: [7 * w // 10, 3 * h // 8],
        645: [9 * w // 10, 3 * h // 8],
        650: [w          , 3 * h // 8]
        }
    return dic

def corners_ordering(pts):
    """Order will be:
    0       1
    3       2
    on an image """
    x_med, y_med = np.median(pts, axis=0)
    left = [(x,y) for x, y in pts if x < x_med]
    right = [(x,y) for x, y in pts if x > x_med]
    left_up = [(x,y) for x, y in left if y < y_med]
    left_down = [(x, y) for x, y in left if y > y_med]
    right_up = [(x, y) for x, y in right if y < y_med]
    right_down = [(x, y) for x, y in right if y > y_med]
    ordered_pts = np.array([left_up[0], right_up[0], right_down[0], left_down[0]])
    return ordered_pts


def change_perspective(img, pts_input, pts_output, shape_output_img):
    homography = cv2.getPerspectiveTransform(pts_input, pts_output)
    transformed_image = cv2.warpPerspective(img, homography, (shape_output_img[0], shape_output_img[1]))
    return transformed_image, homography


def perspective_modifier(pts, img, shape_output_image, situation='start'):
    if type(pts) == list:
        # resize due to the output format (s1, 1, s3)
        pts = np.array(pts)
        pts = np.resize(pts, (pts.shape[0], pts.shape[-1]))

    # order the corners
    pts = corners_ordering(pts)

    # this is where we need to do something with info
    swimpool = swimming_pool_dic(shape_output_image)
    if situation == 'start':
        pts_output = np.array([swimpool[645], swimpool[650], swimpool[450], swimpool[445]])
    elif situation == '5_15' or situation == '15_5':
        pts_output = np.array([swimpool[635], swimpool[645], swimpool[445], swimpool[435]])
    elif situation == '15_25' or situation == '25_15':
        pts_output = np.array([swimpool[625], swimpool[635], swimpool[435], swimpool[425]])
    elif situation == '25_35' or situation == '35_25':
        pts_output = np.array([swimpool[615], swimpool[625], swimpool[425], swimpool[415]])
    elif situation == '35_45' or situation == '45_35':
        pts_output = np.array([swimpool[605], swimpool[615], swimpool[415], swimpool[405]])
    elif situation == '45_50':
        pts_output = np.array([swimpool[600], swimpool[605], swimpool[405], swimpool[400]])

    from_above, hgraphy = change_perspective(img, np.float32(pts), np.float32(pts_output), shape_output_image)
    return from_above, hgraphy, pts

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
    file_name = "3lanes_1_entireimg.jpg"
    img = cv2.imread(path + file_name)
    shape_output_image = (1000, 1000)

    # # for 3lanes_90
    # pts = np.array([[273, 437], [167, 300], [966, 208], [1151,  326]])

    # # for 3 lanes_80
    # pts = np.array([[108,  310], [893,  210], [1085, 322], [216,  444]])

    # for 3lanes_1
    pts = np.array([[385, 355], [660, 308],[555, 449], [832, 399]])

    # ordering in
    # 1        2
    # 4        3
    pts = corners_ordering(pts)

    print(swimming_pool_dic((8, 50)))

    # get the point coordinates of the output image with the knowledge of what we identified
    swimpool = swimming_pool_dic(shape_output_image)
    pts_output = np.array([swimpool[645], swimpool[650], swimpool[450], swimpool[445]])

    print(np.array(pts))
    print(np.array(pts_output))
    from_above, hgraphy = change_perspective(img, np.float32(pts), np.float32(pts_output), shape_output_image)
    print(hgraphy)
    print_image(from_above, title='transformed image')
    # cv2.imwrite('./media/results/successful_homography.jpg', from_above)