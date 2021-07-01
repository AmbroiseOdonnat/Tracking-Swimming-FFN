import numpy as np
import cv2

from color_filtering import color_filter


if __name__ == "__main__":
    ouput_size = (1024, 1024)
    img_objective = cv2.imread('./media/Swimming_pool_50m_above.png')
    img_objective = cv2.resize(img_objective, ouput_size)
    filtered_obj = color_filter(img_objective, colors=['redred'])
    _, mask_obj = cv2.threshold(cv2.cvtColor(filtered_obj, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    cv2.imshow('lol2', mask_obj)


    img_test = cv2.imread('./media/3lanes_80.jpg')
    img_test = cv2.resize(img_test, ouput_size)
    filtered_test = color_filter(img_test, colors=['redred'])
    _, mask_test = cv2.threshold(cv2.cvtColor(filtered_test, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    im_cost = mask_obj*mask_test

    print(np.sum(im_cost))

    # displaying
    # cv2.imshow('lol', filtered_obj)
    # cv2.imshow('lol2', mask_obj)
    # cv2.imshow('lol3', mask_test)
    # cv2.imshow('lol4', im_cost)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # saving
    cv2.imwrite('./media/opti_obj_red.png', mask_obj)
    cv2.imwrite('./media/opti_test_red.png', mask_test)

