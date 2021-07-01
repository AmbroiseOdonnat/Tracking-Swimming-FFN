import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from line_detection import print_image
from color_filtering import color_filter
from line_detection import skeleton

# Load the image
path = "./media/"
file_name = "3lanes.jpg"
img = cv2.imread(path + file_name)
# print_image(img, title='Original')

# for 3lane
lines_lanes = np.array([[[472.66666, 1.4486233]],[[394.33334, 1.4486233]],[[323,1.4486233]]])
lanes_side = np.array([[[472.66666, 1.4486233]], [[323,1.4486233]]])

# # for 3lane_lf
# lines_lanes = np.array([[[260.5, 1.5358897]],  [[418.25,  1.5358897]], [[334., 1.5358897]]])
# lanes_side = np.array([[[260.5, 1.5358897]], [[418.25,  1.5358897]]])

# without the middle line
mask_lines = np.zeros((img.shape[0], img.shape[1]), np.uint8)
for coords in lanes_side:
    rho, theta = coords[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))

    cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 10)
# print_image(mask_lines)
img = cv2.bitwise_and(img, img, mask=mask_lines)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# print_image(laplacian)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
# print_image(laplacian)

filtered = color_filter(img)
edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
# print_image(edges)

# def from_red_to_yellow(hsv):
#     for all the image:
#         if pixel colonne i is jaune and pixel colonne i+1 is red
#             pixel += pixel1, pixel2
#     return transition_pixels
#
# Autre idée: chopper les pixels en histogramme sur leur quantité de rouge ou jaune et de voir les peaks

filtered_red = color_filter(img, ['red'])
skel = skeleton(filtered_red)
print_image(skel)

# # blob detector
# gray_red = cv2.cvtColor(filtered_red, cv2.COLOR_BGR2GRAY)
# gray_red = cv2.bitwise_not(gray_red)
# # print_image(gray_red, title='Gray red')
#
# params = cv2.SimpleBlobDetector_Params()
# # Change thresholds
# params.minThreshold = 1
# params.maxThreshold = 255
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 1
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(gray_red)
# keypoints_coord = np.array([[int(keypoint.pt[0]), int(keypoint.pt[1])] for keypoint in keypoints])
# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # print_image(im_with_keypoints, title='keypoints')
#
# # clustering of the keypoints
# clustering = DBSCAN(eps=100, min_samples=1)
# clustering.fit(keypoints_coord)
# print(clustering.labels_)
# # plt.scatter(keypoints_coord[:, 0], keypoints_coord[:, 1], c=clustering.labels_.astype(float))
# # plt.show()

# # right upper corner, left upper corner, left lower corner, right lower corner
# pts = []
# values, counts = np.unique(clustering.labels_, return_counts=True)
# for i in range(len(values)):
#     value = values[i]
#     count = counts[i]
#     cluster_indexes = np.argwhere(clustering.labels_ == value)
#     cluster = keypoints_coord[cluster_indexes]
#     if count <= 5:
#         cluster_point = np.mean(cluster, axis=0)
#         pts.append(cluster_point.astype(int))
#
# for i in range(len(values)):
#     value = values[i]
#     count = counts[i]
#     cluster_indexes = np.argwhere(clustering.labels_ == value)
#     cluster = keypoints_coord[cluster_indexes]
#     if count > 5:
#         dists = np.sqrt(np.sum((pts[0] - cluster) ** 2, axis=2))
#         print(dists)
#         mindist, minid = dists.min(), dists.argmin()
#         dists2 = np.sqrt(np.sum((pts[1] - cluster) ** 2, axis=2))
#         print(dists2.shape)
#         print(minid)
#         mindist2 = dists2.min()
#         if mindist2 < mindist:
#             minid = dists2.argmin()
#             print(minid)
#         pts.append(cluster[minid])
#
# print(pts)
#
# # when I inverse the gray image of the red
# detected = cv2.circle(img, (314, 437), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (205, 296), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (1215,  323), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (1024,  203), 5, (255,0,0), -1)
# print_image(detected)

# # it seems better when invert the gray
# # not inverted gray:
# detected = cv2.circle(img, (312, 438), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (202, 297), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (1238,  321), 5, (255,0,0), -1)
# detected = cv2.circle(detected, (1036,  202), 5, (255,0,0), -1)
# print_image(detected)
# cv2.imwrite('./media/results/lane_feature_id_first_res.jpg', detected)

# pts = [np.array([[314, 437]]), np.array([[205, 296]]), np.array([[1215,  323]]), np.array([[1024,  203]])]

# # corner detection
# gray_red = np.float32(gray_red)
# dst = cv2.cornerHarris(gray_red,2,3,0.04)
#
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[255,0,0]
#
# print_image(dst, title='Corner detection')
