import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from color_filtering import color_filter

import matplotlib
matplotlib.use('TkAgg')

def skeleton(img):

    if img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = edges = gray.copy()
    else:
        edges = img.copy()
    # edges = cv2.Laplacian(gray, cv2.CV_8UC1)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Threshold the image
    ret, img = cv2.threshold(edges, 127, 255, 0)

    # print_image(edges, title='edges')

    # Step 1: Create an empty skeleton
    skel = np.zeros(edges.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(edges, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(edges, element)
        skel = cv2.bitwise_or(skel,temp)
        edges = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(edges)==0:
            break
    return skel

def hough_line_mask(img, threshold_hough=120, on_img=None):
    if img.shape[-1] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    else:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 360., threshold_hough)

    clustered_lines = clustering_lines(lines, 0.01, 3)
    clustered_lines = get_yellow_lines(clustered_lines, img)

    if on_img is None:
        mask_lines = np.zeros(edges.shape, np.uint8)
    else:
        mask_lines = on_img



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

def clustering_lines(lines, eps, min_samples):
    # A low minPts means it will build more clusters from noise, so don't choose it too small.

    # shaping of the line to enter the model)
    lines = np.array(lines)
    # plt.plot(lines[:, 0, 0], lines[:, 0, 1], '*')
    # plt.ylabel('Theta')
    # plt.show()
    # print('Number of lines found', len(lines))

    # fitting
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(lines[:, :, 1])

    # identification of the biggest cluster
    values, counts = np.unique(clustering.labels_, return_counts=True)
    ind = np.argmax(counts)
    big_clus_numb = values[ind]


    # extraction of the lines of the biggest cluster
    biggest_cluster_indexes = np.argwhere(clustering.labels_ == big_clus_numb)
    biggest_cluster = lines[biggest_cluster_indexes]
    biggest_cluster = np.resize(biggest_cluster, (biggest_cluster.shape[0], 1, biggest_cluster.shape[-1])) #shape of lines from hough transform
    # print('Total number of lines before rho clustering', len(biggest_cluster))

    # then clustering on the rho (we should have 11 clusters)
    clustering_rho = DBSCAN(eps=20, min_samples=1)
    clustering_rho.fit(biggest_cluster[:, :, 0])
    # plt.scatter(biggest_cluster[:, 0, 0], biggest_cluster[:, 0, 1], c=clustering_rho.labels_.astype(float))
    # plt.show()

    # we get by clusters and we get the mean of each cluster
    lanes = []
    values, counts = np.unique(clustering_rho.labels_, return_counts=True)
    for i in range(len(values)):
        value = values[i]
        count = counts[i]
        cluster_indexes = np.argwhere(clustering_rho.labels_ == value)
        cluster = biggest_cluster[cluster_indexes]
        # print(cluster.shape, i)
        cluster = np.mean(cluster, axis=0)
        lanes.append(cluster)
    lanes = [lane[0] for lane in lanes]
    lanes = np.array(lanes)
    # print('Number of cluster', len(values))
    # print('Number of lanes', len(lanes))
    return lanes

def foreground_extraction(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # Do masking
    res = cv2.bitwise_and(img, img, mask=mask)
    # convert to hsv to gray
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Defining a kernel to do morphological operation in threshold #image to get better output.
    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(res_gray, 50, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh

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

def frame_line_extraction(img, lower, upper, threshold):
    # foreground extraction
    extracted = foreground_extraction(img, lower, upper)

    # resizing
    target_size = (500, 500)
    extracted = cv2.resize(extracted, target_size)

    # hough lines computation
    mask_lines, _ = hough_line_mask(extracted, threshold_hough=threshold)

    # get the skeleton
    skel = skeleton(extracted)

    # fusion of the skeleton and the lines
    mask_complete = cv2.bitwise_and(skel, skel, mask=mask_lines)

    return mask_complete

def frame_color_filter_and_hough(img, threshold):
    filtered = color_filter(img, colors=['yellow', 'red'])
    # print_image(filtered)
    mask_lines, lines = hough_line_mask(filtered, threshold_hough=threshold)
    # print_image(mask_lines)
    res = cv2.bitwise_and(img, img, mask=mask_lines)
    return res, lines

# utils
def print_image(img, title='figure'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.show(block=True)

if __name__ == "__main__":

    # Load the image and resize
    path = "./media/"
    file_name = "3lanes_1_entireimg.jpg"
    img = cv2.imread(path + file_name)

    # print_image(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    # # for brasse_50.jpg
    # lower = np.array([90, 150, 100])
    # upper = np.array([100, 255, 255])
    # threshold =  120

    # # for titenis.jpg
    # lower = np.array([90, 130, 100])
    # upper = np.array([105, 255, 255])
    # threshold = 120
    #
    # # # foreground extraction
    # # extracted = foreground_extraction(img, lower, upper)
    # # # print_image(extracted, title="Extracted from the background")
    # #
    # # # # resizing
    # # # target_size = (500, 500)
    # # # extracted = cv2.resize(extracted, target_size)
    # #
    # # # get the skeleton
    # # skel = skeleton(extracted)
    # # print_image(skel, title='Skeleton')
    #
    # # Test using the edges of canny instead of a skeletons
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # skel = cv2.Canny(gray, 50, 150, apertureSize=3)
    #
    # # then hough lines on the filtered image
    # filtered = color_filter(img)
    #
    # # hough lines computation
    # mask_lines, _ = hough_line_mask(filtered, threshold_hough=threshold) # on_img=cv2.resize(img, target_size))
    # print_image(mask_lines, title="hough lines mask")
    # #
    # # # fusion of the skeleton and the lines
    # # mask_complete = cv2.bitwise_and(skel, skel, mask = mask_lines)
    # # print_image(mask_complete, title='mask complete')
    # #
    # # add the colors
    # # resize_frame = cv2.resize(img, (500, 500))
    # resize_frame = img.copy()
    # # res = cv2.bitwise_and(resize_frame, resize_frame, mask=mask_complete)
    # res = cv2.bitwise_and(resize_frame, resize_frame, mask=mask_lines)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # print_image(res, title='Mask applied on original image')



    # The whole procedure
    mask_complete, _ = frame_color_filter_and_hough(img, 100)
    print_image(mask_complete, title='mask complete')
    # cv2.imwrite('./media/start_allines.png', mask_complete)

    ## idea: définir un range de theta pour éviter d'avoir d'autres gros clusters qui se manifestent

