import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

path = "./media/50_brasse_short.avi"
cap = cv2.VideoCapture(path)
minLineLength = 100
maxLineGap = 10
target_size = (500, 500)
# black = np.zeros(target_size)
ret, frame = cap.read()
frame0 = np.copy(frame)
frame0 = cv2.resize(frame0, target_size)
while ret :
    frame = cv2.resize(frame, target_size)
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_8UC1)
    lines = cv2.HoughLinesP(laplacian, 1, np.pi/180, 1, minLineLength, maxLineGap)
    for coords in lines:
        x1, y1, x2, y2 = coords[0]
        cv2.line(frame0, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = cv2.HoughLines(laplacian, 1, np.pi/180, 250)
#     if len(lines) > 0 :
#         for machins in lines :
#             rho, theta = machins[0]
#             if theta < np.pi*13/24 and theta > np.pi*11/24 :
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a*rho
#                 y0 = b*rho
#                 x1 = int(x0 + 1000*(-b))
#                 y1 = int(y0 + 1000*(a))
#                 x2 = int(x0 - 1000*(-b))
#                 y2 = int(y0 - 1000*(a))
#                 cv2.line(frame,(x1,y1),(x2,y2),(0,0,0),1)
        ret, frame = cap.read()
frame0 = cv2.resize(frame0, (1000, 1000))
plt.imshow(frame0)
plt.title("toto")
plt.show(block=True)
cap.release()