# Script for computing the 3 middle lines of an image of a swimming pool with olympic lanes

import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import hsv_mask

def color_filter(img, colors=['yellow', 'red']):
    """Color filtering uses the dictionary of colors in HSV defined in hsv_mask.py to filter an image by colors"""

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = hsv_mask.create_mask(img_hsv, colors)
    masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask=masks)
    return cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)

def hough_line_mask(img, threshold_hough=190):
    """Input: image BGR, a threshold used for the hough transform
    Output: the mask for the lines, a list of the lines [rho, theta] of shape (n, 1, 2)

     This function uses the hough transform to compute the lines. They are then clustered (see clustered_lines).
     Then the lines are translated into a mask that can be applied to an image."""

    if img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    else:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 360., threshold_hough)

    clustered_lines = clustering_lines(lines)
    clustered_lines = get_yellow_lines(clustered_lines, img)

    mask_lines = np.zeros(edges.shape, np.uint8)
    for coords in clustered_lines:
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
    return mask_lines, clustered_lines




def clustering_lines(lines):
    """Input: lines
    Output: clustered lines
    We firstly cluster by theta to get the three middle lines.
    Secondly, we cluster by rho to group lines that represents the same line
    on the image and take the median of thetas.
    Problems that could occur: bad detection of the lines = the clustering on theta will not give the three middle
    lines as the biggest cluster.
    2 prob: detection too good of the lines: we detected the whole swimming pool"""
    # A low minPts means it will build more clusters from noise, so don't choose it too small.

    # shaping of the line to enter the model)
    lines = np.array(lines)

    # fitting on theta
    clustering = DBSCAN(eps=0.01, min_samples=3)
    clustering.fit(lines[:, :, 1])

    # identification of the biggest cluster with the theta not too big
    values, counts = np.unique(clustering.labels_, return_counts=True)
    max_count = 0
    big_clus_numb = values[0]
    for value, count in zip(values, counts):
        indexes = np.argwhere(clustering.labels_ == value)
        current_cluster = lines[indexes]
        if count > max_count and np.max(current_cluster[:, :, :,  1]) < 1.9 and np.max(current_cluster[:, :, :,  1]) > 1.25:
            big_clus_numb = value
            max_count = count
    # ind = np.argmax(counts)
    # big_clus_numb = values[ind]


    # extraction of the lines of the biggest cluster
    biggest_cluster_indexes = np.argwhere(clustering.labels_ == big_clus_numb)
    biggest_cluster = lines[biggest_cluster_indexes]
    biggest_cluster = np.resize(biggest_cluster, (biggest_cluster.shape[0], 1, biggest_cluster.shape[-1]))

    # then clustering on the rho (we should have 3 clusters)
    clustering_rho = DBSCAN(eps=20, min_samples=1)
    clustering_rho.fit(biggest_cluster[:, :, 0])

    # we get by clusters and we get the mean of each cluster
    lanes = []
    values, counts = np.unique(clustering_rho.labels_, return_counts=True)
    for i in range(len(values)):
        value = values[i]
        count = counts[i]
        cluster_indexes = np.argwhere(clustering_rho.labels_ == value)
        cluster = biggest_cluster[cluster_indexes]
        cluster = np.mean(cluster, axis=0)
        lanes.append(cluster)
    lanes = [lane[0] for lane in lanes]
    lanes = np.array(lanes)

    return lanes

def get_yellow_lines(lines, img):
    """Returns the lines that contains yellow the most yellow"""
    if len(lines) <= 3:
        return lines
    filtered_frame = color_filter(img, colors=['yellow'])
    number_ye_pix = []
    for i, coords in enumerate(lines):
        rho, theta = coords[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))

        one_line_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.line(one_line_mask, (x1, y1), (x2, y2), 255, 10)
        one_line = cv2.bitwise_and(filtered_frame, filtered_frame, mask=one_line_mask)
        one_line = cv2.cvtColor(one_line, cv2.COLOR_BGR2GRAY)
        number_ye_pix.append([i, np.sum(one_line[one_line.astype(bool)])])
    number_ye_pix = sorted(number_ye_pix, key=lambda x: x[1], reverse=True)
    ye_lines = [lines[number_ye_pix[0][0]], lines[number_ye_pix[1][0]], lines[number_ye_pix[2][0]]]
    return np.array(ye_lines)

def frame_color_filter_and_hough(img, threshold):
    """Input: an image in BGR
       Output: a masked image with the main lines in BGR, a list of the lines [rho, theta] of shape (n, 1, 2)

    This function computes the mask for the main line on an image the has been filtered by red and yellow.
    it is basically made to dectect the lines in swimming pools with olympic settings
    First we filter the colors and then we compute the hough lines and cluster them.
    The threshold is used in the hough transform. The higher, the less lines are detected.
    However, if too many lines are detected we might end up have more then just the 3 middle lines.
    For the moment, and after some tests, it is set to 190."""

    filtered = color_filter(img)
    mask_lines, lines = hough_line_mask(filtered, threshold_hough=threshold)
    res = cv2.bitwise_and(img, img, mask=mask_lines)
    return res, lines

def well_detected_lines(lines):
    """Input: lines
    Output: a boolean
    returns True if the the lines follows some condition that we choose in the function

    For the moment, the conditions are related to the detection of blob. That is to say, we need to see on ly three
    lines, and to be almost sure that we have the lines of the swimming pool they should be horizontal"""

    return len(lines) == 3 and np.max(lines[:, :, 1]) < 1.8
if __name__ == "__main__":
    path = "./media/"
    file_name = "3lanes_91_entireimg.jpg"
    img = cv2.imread(path + file_name)
    res, lines = frame_color_filter_and_hough(img, 190)
    print(len(lines) == 3)
    cv2.imshow('Test', res)
    cv2.waitKey(0)