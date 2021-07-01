import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import math
import sys

path = "./media/"
file_name = "brasse_50.jpg"

image = cv2.imread(path + file_name)
target_size = (500, 500)
image = cv2.resize(image, target_size)

# alpha = int(sys.argv[1])
# beta = int(sys.argv[2])
# image = image*alpha + beta
#
# plt.figure(figsize=(8, 6))
# plt.imshow(image)
# plt.show(block=True)

#
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

plt.figure(figsize=(8, 6))
plt.imshow(edges)
plt.show(block=True)

lines = cv2.HoughLines(edges, 1, np.pi/180, 250)
print(lines.shape)

image2 = image.copy()
for coords in lines:
    rho, theta = coords[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image2,(x1,y1),(x2,y2),(0,0,255),2)

# image2 = image.copy()
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 1, minLineLength, maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#      cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)

plt.figure(figsize=(8, 6))
plt.imshow(image2)
plt.show(block=True)


# plt.imshow(image)
# plt.show(block=True)
#
# blue, green, red = cv2.split(image)
# plt.figure(figsize=(8,6))
# plt.imshow(red, cmap=plt.cm.gray)
# plt.show(block=True)
#
# plt.figure(figsize=(8,6))
# plt.imshow(blue, cmap=plt.cm.gray)
# plt.show(block=True)
#
# plt.figure(figsize=(8,6))
# plt.imshow(green, cmap=plt.cm.gray)
# plt.show(block=True)







#
# img = cv2.imread(path + file_name, 0)
#
# # Threshold the image
# ret,img = cv2.threshold(img, 127, 255, 0)
#
# # Step 1: Create an empty skeleton
# size = np.size(img)
# skel = np.zeros(img.shape, np.uint8)
#
# # Get a Cross Shaped Kernel
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
#
# # Repeat steps 2-4
# while True:
#     #Step 2: Open the image
#     open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
#     #Step 3: Substract open from the original image
#     temp = cv2.subtract(img, open)
#     #Step 4: Erode the original image and refine the skeleton
#     eroded = cv2.erode(img, element)
#     skel = cv2.bitwise_or(skel,temp)
#     img = eroded.copy()
#     # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
#     if cv2.countNonZero(img)==0:
#         break
#
# # Displaying the final skeleton
# plt.figure(figsize=(12, 10))
# plt.imshow(skel)
# plt.show(block=True)
#
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(skel, 1, np.pi/180, 1, minLineLength, maxLineGap)
#
# image2 = image.copy()
# for coords in lines:
#      x1,y1,x2,y2 = coords[0]
#      cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)
# print(lines.shape)
#
# # lines = cv2.HoughLines(skel, 1, np.pi/180, 250)
# # print(lines.shape)
#
# # image2 = image.copy()
# # for coords in lines:
# #     rho, theta = coords[0]
# #     a = np.cos(theta)
# #     b = np.sin(theta)
# #     x0 = a*rho
# #     y0 = b*rho
# #     x1 = int(x0 + 1000*(-b))
# #     y1 = int(y0 + 1000*(a))
# #     x2 = int(x0 - 1000*(-b))
# #     y2 = int(y0 - 1000*(a))
# #
# #     cv2.line(image2,(x1,y1),(x2,y2),(0,0,255),2)
#
#
#
# plt.figure(figsize=(12, 10))
# plt.imshow(image2)
# plt.show(block=True)
