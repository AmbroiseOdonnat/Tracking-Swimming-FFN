import numpy as np
import cv2
import matplotlib.pyplot as plt

import hsv_mask


def color_filter(img, colors=['red', 'yellow']):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = hsv_mask.create_mask(img_hsv, colors)
    masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=masks)
    return cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':

    from line_detection import print_image
    path = "./media/"
    file_name = "halfspa.jpg"
    img = cv2.imread(path + file_name)

    # colors to filter on
    colors = ['red', 'yellow']

    # filtering
    filtered = color_filter(img, colors=colors)
    print_image(filtered, title='Masked image')


    # res_inv = cv2.bitwise_not(img)
    # # look for the inverse of red which is cyan
    # # technique explaines in https://stackoverflow.com/questions/32522989/opencv-better-detection-of-red-color
    # lower_cyan = np.array([80, 70, 50])
    # upper_cyan = np.array([100, 255, 255])
    # hsv = cv2.cvtColor(res_inv, cv2.COLOR_BGR2HSV)
    # print_image(hsv, title='HSV')
    # mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    # print_image(mask, title='Mask red')
    # only_red = cv2.bitwise_and(img, img, mask=mask)
    # print_image(cv2.cvtColor(only_red, cv2.COLOR_BGR2RGB), title='Final print of the lines')

    cv2.imwrite('./media/halfspa_ry.jpg', cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
