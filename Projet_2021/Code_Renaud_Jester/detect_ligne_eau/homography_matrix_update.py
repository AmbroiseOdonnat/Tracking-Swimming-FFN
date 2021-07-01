import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from blop_detection_lane import draw_lines_on_image
from line_detection import print_image

def point_update(x1, y1, rho1, theta1, rho2, theta2):
    # theta modification by pi/2
    theta1 = np.pi/2 - theta1
    theta2 = np.pi/2 - theta2
    Na = math.sqrt(x1**2 + (rho1 - y1)**2)
    deltar = rho2 - rho1
    cos1 = math.cos(theta1)
    cos2 = math.cos(theta2)
    sin1 = math.sin(theta1)
    sin2 = math.sin(theta2)
    cos22 = cos2**2
    sin22 = sin2**2
    a = 2*(Na*cos1 - x1)
    c = (Na*cos1 - x1)**2 - (rho2**2)*cos22
    d = 2*(Na*sin1 - y1 - deltar + rho2*sin22)
    e =  (Na*sin1 - y1 - deltar)**2 - (rho2**2)*sin22
    dpr = d + 2*rho2*cos22
    f = -dpr/a
    g = -(c + e)/a

    # second degree equation
    asec = cos22 - sin22*(f**2)
    b = d - 2*f*g*sin22
    csec = e - sin22*(g**2)
    deltasec = b**2 - 4*asec*csec
    try:
        sol1_y = (- b - math.sqrt(b**2 - 4*asec*csec))/(2*asec)
        sol2_y = (- b + math.sqrt(b**2 - 4*asec*csec))/(2*asec)
    except ValueError:
        print("Impossble de calcular el delta")
        return -100000, -100000

    sol1_x = f*sol1_y + g
    sol2_x = f*sol2_y + g

    # need to choose what is the solution maybe a better solution would be to check if they are on the line
    if sol2_x < 0 or sol2_y < 0:
        x2 = sol1_x
        y2 = sol1_y
    elif sol1_x < 0 or sol1_y < 0:
        x2 = sol2_x
        y2 = sol2_y
    else:
        print('Problème')

    if x2 < 0 or y2 < 0:
        print("Probleme")
    # print("Sol1: ", int(sol1_x), int(sol1_y))
    # print("Sol2: ",int(sol2_x), int(sol2_y))
    # print()
    return round(x2), round(y2)

if __name__ == "__main__":
    path = "./media/"
    file_name = "3lanes_91.jpg"
    img = cv2.imread(path + file_name)

    # 3 lanes 80
    lines_lanes = np.array([[[322.7143, 1.4461299]], [[469.75, 1.4333516]], [[393.5, 1.4384421]]])
    lane80 = [322.7143, 1.4461299]
    from blop_detection_lane import select_outside_lines
    lanes_side = select_outside_lines(lines_lanes)
    print(lanes_side)
    # take [893, 210] upper right
    pts80 = np.array([np.array([216, 444]), np.array([108, 310]), np.array([893, 210]), np.array([1085, 322])])
    xy80 = [893, 210]
    from perspective_modifier import corners_ordering
    print(corners_ordering(pts80))
    # 3 lanes_90
    lines_lanes = np.array([[[473.7143 , 1.4398965]], [[394.75, 1.4453506]], [[319.25, 1.4551682]]])
    lane90 = [319.25, 1.4551682]
    # lanes_side = select_outside_lines(lines_lanes)
    # take [969, 207] upper right
    pts90 = [np.array([[273, 436]]), np.array([[166, 302]]), np.array([[969, 207]]), np.array([[1159,  324]])]
    xy90 = [969, 207]

    # 3 lanes_91
    lines_lanes = np.array([[[474.33334, 1.4389269]], [[393.16666, 1.4471687]], [[318.33334, 1.4563804]]])
    lane91 = [318.33334, 1.4563804]
    # lanes_side = select_outside_lines(lines_lanes)
    # take [976, 208] upper right on the upper lane
    pts91 = [np.array([[282, 438]]), np.array([[171, 301]]), np.array([[976, 208]]), np.array([[1163,  324]])]
    xy91 = [976, 208]


    # img = draw_lines_on_image(img, [[[318.33334, 1.4563804]]])
    # img = draw_lines_on_image(img, [[[318.33334, np.pi/2]]])
    # print_image(img, title='outside lines')


    x1, y1 = (200, 500) #xy80
    rho1, theta1 = lane80
    rho2, theta2 = lane80

    x2, y2 = point_update(x1, y1, rho1, theta1, rho2, theta2)
    print(x2, y2)
    print("Solt: ", xy90[0], xy90[1])