# code from: https://medium.com/@kananvyas/player-and-football-detection-using-opencv-python-in-fifa-match-6fd2e4e373f0
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

import math
import sys


def fine_color_range(coarse, around_mean=0.3):
    lower = []
    upper = []
    for i in range(3):
        hist = cv2.calcHist([coarse], [i], None, [256], [0, 256])
        data = hist[1:]  # get rid of the zeros
        data = np.reshape(data, (len(data),))
        # print(np.reshape(data, (len(data),)).shape)
        # print(np.arange(1, len(data)+1).shape)
        mean = np.average(np.arange(1, len(data)+1), weights=data[:])
        lower_bound = int(mean - around_mean*mean)
        upper_bound = int(around_mean*mean + mean)
        lower.append(lower_bound)
        upper.append(upper_bound)
    return np.array(lower), np.array(upper)


path = "./media/"
file_name = "raph_swimming_pool.jpg"
image = cv2.imread(path + file_name)
plt.imshow(image)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

lower_blue = np.array([0, 80, 80])
upper_blue = np.array([200, 120, 120])

mask = cv2.inRange(lab, lower_blue, upper_blue)
print(np.unique(mask), type(mask))
res = cv2.bitwise_and(lab, lab, mask=mask)

# hist = cv2.calcHist([res],[2],None,[256],[0,256])
# plt.plot(hist[1:])
plt.imshow(res)
plt.show()

fine_lower, fine_upper = fine_color_range(res)
print(fine_lower, fine_upper)
fine_mask = cv2.inRange(lab, fine_lower, fine_upper)
plt.imshow(fine_mask)
plt.show()

# image = cv2.imread(path + file_name)
# #converting into hsv image
# hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#  #green range
# lower_green = np.array([40,40, 40])
# upper_green = np.array([70, 255, 255])
#  #blue range
# lower_blue = np.array([90,150,100])
# upper_blue = np.array([100,255,255])
# #Red range
# lower_red = np.array([0,31,255])
# upper_red = np.array([176,255,255])
# #white range
# lower_white = np.array([0,0,0])
# upper_white = np.array([0,0,255])
#
# # By hand color selection
# # To do: Test on different type of image. Make it more robust?
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# #Do masking
# res = cv2.bitwise_and(image, image, mask=mask)
# #convert to hsv to gray
# res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
# res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
#
# plt.subplot(121), plt.imshow(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)), plt.title('RGB')
# plt.subplot(122), plt.imshow(res_gray), plt.title('Gray')
# plt.show(block=True)
#
# #Defining a kernel to do morphological operation in threshold #image to get better output.
# kernel = np.ones((13,13),np.uint8)
# thresh = cv2.threshold(res_gray,50,255,cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
# plt.imshow(thresh)
# plt.show(block=True)
#
#
# target_size = (500, 500)
# thresh = cv2.resize(thresh, target_size)
#
# # edges = cv2.Canny(thresh, 50, 150, apertureSize = 3)
# edges = cv2.Laplacian(thresh, cv2.CV_8UC1)
# plt.imshow(edges)
# plt.show(block=True)
#
#
#
# lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
# print(lines.shape)
#
# image2 = image.copy()
# image2 = cv2.resize(image2, target_size)
# for coords in lines:
#     rho, theta = coords[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(image2,(x1,y1),(x2,y2),(0,0,255),2)
#
# # image2 = image.copy()
# # minLineLength = 10
# # maxLineGap = 10
# # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 1, minLineLength, maxLineGap)
# # for x1,y1,x2,y2 in lines[0]:
# #      cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)
#
# plt.imshow(image2)
# plt.show(block=True)
#
# mask_lines = np.zeros(edges.shape, np.uint8)
# for coords in lines:
#     rho, theta = coords[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(mask_lines,(x1,y1),(x2,y2),255,2)
#
# fusion = cv2.bitwise_and(edges, edges, mask = mask_lines)
# image3 = image.copy()
# image3 = cv2.resize(image3, target_size)
# fusion = cv2.bitwise_and(image3, image3, mask = fusion)
# plt.imshow(fusion)
# plt.show(block=True)
