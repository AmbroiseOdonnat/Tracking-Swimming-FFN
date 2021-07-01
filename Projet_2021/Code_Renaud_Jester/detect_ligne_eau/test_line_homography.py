# implementation based on
# https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=homography+estimation+elan+dubrofsky&btnG=&oq=homography+estimation+elan#

import cv2
import numpy as np
from math import cos, sin, pi
import scipy

from blop_detection_lane import select_outside_lines
from perspective_modifier import swimming_pool_dic, change_perspective, corners_ordering


def point_matrix_computing(pt_origin, pt_destination):
    x, y = pt_origin
    u, v = pt_destination
    A = np.zeros((2, 9))
    A[0, 0], A[0, 1] = -x, -y
    A[0, 2] = -1
    A[1, 3], A[1, 4] = -x, -y
    A[1, 5] = -1
    A[0, 6:9] = u * np.array([x, y, 1])
    A[1, 6:9] = v * np.array([x, y, 1])
    return A


def from_angle_to_cartesian(line):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 100 * (-b))
    y1 = int(y0 + 100 * (a))
    x2 = int(x0 - 100 * (-b))
    y2 = int(y0 - 100 * (a))
    # TODO handle the case where rho is zero
    return from_two_points_to_cartesian([x1, y1], [x2, y2])


def from_two_points_to_cartesian(pta, ptb):
    a, b = pta
    c, d = ptb
    coef_norm = -(c - a)*b + a*(d - b)
    e = -(d-b)/coef_norm
    f = (c-a)/coef_norm
    return [e, f]


def line_matrix_computing(line_origin, line_dest):
    x, y = line_origin
    u, v = line_dest
    A = np.zeros((2, 9))
    A[0, 0], A[1, 1] = -u, -u
    A[0, 2], A[1, 2] = u * np.array([x, y])
    A[0, 3], A[1, 4] = -v, -v
    A[0, 5], A[1, 5] = v * np.array([x, y])
    A[0, 6], A[1, 7] = -1, -1
    A[0, 8], A[1, 8] = 1 * np.array([x, y])
    return A


def compute_hm_from_Ais(Ais):
    # b = np.zeros((Ais.shape[0], ))
    # b[-1] = 1
    # h_coef = np.linalg.solve(Ais, b)
    # np.append(h_coef, 1)
    # hm = np.reshape(h_coef, (3, 3))
    # hm = np.linalg.lstsq(Ais, np.zeros((Ais.shape[0], )))
    h_coef = scipy.linalg.null_space(Ais)
    # h_coef = np.append(h_coef, 1)
    print(h_coef)
    hm = np.reshape(h_coef, (3, 3))
    hm = hm / hm[2, 2] # normalization
    return hm


def apply_homography(img, hm, shape_output_image):
    changed_img = cv2.warpPerspective(img, hm, shape_output_image)
    return changed_img


if __name__ == "__main__":
    img = "./media/3lanes_80_entireimg.jpg"
    img = cv2.imread(img)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    shape_output_image = (1024, 1024)

    swimpool = swimming_pool_dic(shape_output_image)

    # for 3lanes_80
    lines_lanes = np.array([[[322.7143, 1.4461299]], [[469.75, 1.4333516]], [[393.5, 1.4384421]]])
    lanes_side = select_outside_lines(lines_lanes)

    # other points to see if it is a problem that the points are on the same line
    # pt5 = [969, 47]
    # pt_out5 = [1024, 0]
    pt5 = [24, 207]  # 35m mark for the first lane up the image
    pt_out5 = [7 * 1024 // 10, 1024//8]
    pt6 = [436, 714]  # 35m mark for the last lane at the bottom of the image
    pt_out6 = [7 * 1024 // 10, 1024]

    # pts are ordered
    # pts = np.array([[108, 310], [893, 210], [1085, 322], [216, 444]])
    pts = np.array([pt5, [893, 210], [1085, 322], pt6])
    # pts_output = np.array([swimpool[635], swimpool[645], swimpool[445], swimpool[435]])
    pts_output = np.array([pt_out5, swimpool[645], swimpool[445], pt_out6])

    # line_output_6 = [0, 1/(swimpool[635][1])]
    # line_output_4 = [0, 1/(swimpool[435][1])]
    line_output_6 = from_two_points_to_cartesian(swimpool[635], swimpool[645])
    line_output_4 = from_two_points_to_cartesian(swimpool[435], swimpool[445])
    line_output_5 = from_two_points_to_cartesian(swimpool[535], swimpool[545])

    from_above, hgraphy = change_perspective(img, np.float32(pts), np.float32(pts_output), shape_output_image)
    cv2.namedWindow("goal", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("goal", 1024, 1024)
    cv2.imshow('goal', from_above)
    cv2.waitKey(0)

    # print(point_matrix_computing(pts[0], pts_output[0]))
    # cart_line = from_angle_to_cartesian(lines_lanes[0, 0])
    # print(cart_line)
    # print(line_matrix_computing(cart_line, line_output_6))
    line_per6 = from_two_points_to_cartesian([109, 309], [217, 443])
    line_per4 = from_two_points_to_cartesian([892, 215], [1073, 329])
    line_per6_out = from_two_points_to_cartesian(swimpool[435], swimpool[635])
    line_per4_out = from_two_points_to_cartesian(swimpool[645], swimpool[445])
    # print('regarde ici')
    # print(line_per4)
    # print(line_per6)
    print(line_per4_out)
    print(line_per6_out)


    # A1 = point_matrix_computing(pts[0], pts_output[0])
    # A2 = point_matrix_computing(pts[3], pts_output[3])
    # A1 = point_matrix_computing(pt5, pt_out5)
    # A2 = point_matrix_computing(pt6, pt_out6)
    A3 = point_matrix_computing(pts[1], pts_output[1])
    # A3 = point_matrix_computing(pts[2], pts_output[2])

    # A1 = line_matrix_computing(line_per4, line_per4_out)
    A1 = line_matrix_computing(line_per6, line_per6_out)
    cart_line1 = from_angle_to_cartesian(lines_lanes[0, 0])  # line 6
    print(cart_line1)
    cart_line2 = from_angle_to_cartesian(lines_lanes[1, 0])  # line 4
    print(cart_line2)
    cart_line5 = from_angle_to_cartesian(lines_lanes[2, 0])
    # A3 = line_matrix_computing(cart_line1, line_output_6)
    A4 = line_matrix_computing(cart_line2, line_output_4)
    A2 = line_matrix_computing(cart_line5, line_output_5)

    A12 = np.append(A1, A2, axis=0)
    A34 = np.append(A3, A4, axis=0)
    Ais = np.append(A12, A34, axis=0)
    # Ais = np.append(Ais, [[0, 0, 0, 0, 0, 0, 0, 0, 1]], axis=0)
    print_mat = np.zeros((8, 9))
    print_mat[np.where(Ais != 0)] = 1
    print(print_mat)
    hm = compute_hm_from_Ais(Ais)
    # hm[2,2] = 1

    print(hm)

    modified = apply_homography(img, hm, shape_output_image)
    cv2.namedWindow("lol", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("lol", 1024, 1024)
    cv2.imshow('lol', modified)
    cv2.waitKey(0)

    # line = cv2.line(img, (109, 309), (217, 443), 255, 10)
    # line = cv2.line(line, (892, 215), (1073, 329), 255, 10)
    # rho, theta = lines_lanes[0, 0]
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # x1 = int(x0 + 3000 * (-b))
    # y1 = int(y0 + 3000 * (a))
    # x2 = int(x0 - 3000 * (-b))
    # y2 = int(y0 - 3000 * (a))
    # line = cv2.line(line, (x1, y1), (x2, y2), 255, 10)
    # rho, theta = lines_lanes[1, 0]
    # a = np.cos(theta)
    # b = np.sin(theta)
    # x0 = a * rho
    # y0 = b * rho
    # x1 = int(x0 + 3000 * (-b))
    # y1 = int(y0 + 3000 * (a))
    # x2 = int(x0 - 3000 * (-b))
    # y2 = int(y0 - 3000 * (a))
    # line = cv2.line(line, (x1, y1), (x2, y2), 255, 10)

