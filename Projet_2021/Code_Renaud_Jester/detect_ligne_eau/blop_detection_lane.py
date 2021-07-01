import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from line_detection_frame import color_filter

def select_outside_lines(lines):
    """we will have the result in this spatial order:
    0 (upper line index 0)
    1 (lower line index 1)
    """
    rho_lines = np.array([coords[0][0] for coords in lines])

    # index of the max and the min
    indmax = np.argmax(rho_lines)
    indmin = np.argmin(rho_lines)

    outside_lines = np.array([lines[indmin], lines[indmax]])
    return outside_lines

def draw_lines_on_image(img, lines, thickness=10):
    mask_lines = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for coords in lines:
        rho, theta = coords[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 3000 * (-b))
        y1 = int(y0 + 3000 * (a))
        x2 = int(x0 - 3000 * (-b))
        y2 = int(y0 - 3000 * (a))

        cv2.line(mask_lines, (x1, y1), (x2, y2), 255, thickness)
    new_img = cv2.bitwise_and(img, img, mask=mask_lines)
    return new_img

def blop_detection_on_red(img):
    # filtering by red
    filtered = color_filter(img, ['redred'])
    # print_image(filtered, title="Only red")
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # results seems better when invert the gray
    # ret, gray = cv2.threshold(gray, 250, 255, 0)
    # print_image(gray, title='Noir et blanc = gris')

    # threshold pour mettre en noir + faire dilatation + Otsu pour trouver meilleur seuil

    # blop detection
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 100

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    keypoints_coord = np.array([[int(keypoint.pt[0]), int(keypoint.pt[1])] for keypoint in keypoints])
    # im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # print_image(im_with_keypoints, title='All keypoints')
    # print(keypoints_coord)
    return keypoints_coord

def blob_selection_5_15(keypoints_as_coord):
    # clustering of the keypoints
    clustering = DBSCAN(eps=100, min_samples=1) #ça dépend de la video l'eps
    clustering.fit(keypoints_as_coord)
    # print(clustering.labels_)
    # plt.scatter(keypoints_as_coord[:, 0], keypoints_as_coord[:, 1], c=clustering.labels_.astype(float))
    # plt.show()

    # right upper corner, left upper corner, left lower corner, right lower corner
    pts = []
    values, counts = np.unique(clustering.labels_, return_counts=True)
    for i in range(len(values)):
        value = values[i]
        count = counts[i]
        cluster_indexes = np.argwhere(clustering.labels_ == value)
        cluster = keypoints_as_coord[cluster_indexes]
        if count <= 5:
            cluster_point = np.mean(cluster, axis=0)
            pts.append(cluster_point.astype(int))

    for i in range(len(values)):
        value = values[i]
        count = counts[i]
        cluster_indexes = np.argwhere(clustering.labels_ == value)
        cluster = keypoints_as_coord[cluster_indexes]
        if count > 5:
            dists = np.sqrt(np.sum((pts[0] - cluster) ** 2, axis=2))
            mindist, minid = dists.min(), dists.argmin()
            dists2 = np.sqrt(np.sum((pts[1] - cluster) ** 2, axis=2))
            mindist2 = dists2.min()
            if mindist2 < mindist:
                minid = dists2.argmin()
            pts.append(cluster[minid])

    return pts

def blob_selection_start(keypoints_as_coord):
    # clustering of the keypoints
    clustering = DBSCAN(eps=40, min_samples=2)  # ça dépend de la video l'eps
    clustering.fit(keypoints_as_coord)
    # print(clustering.labels_)
    # plt.scatter(keypoints_as_coord[:, 0], keypoints_as_coord[:, 1], c=clustering.labels_.astype(float))
    # plt.show()

    pts = []
    values, counts = np.unique(clustering.labels_, return_counts=True)

    # we select the two biggest cluster
    max, secondmax = np.argsort(counts)[-2:]
    first_cluster = keypoints_as_coord[np.argwhere(clustering.labels_ == values[max])]
    second_cluster = keypoints_as_coord[np.argwhere(clustering.labels_ == values[secondmax])]

    min_x_first = np.argmin(first_cluster[:, 0, 0])
    max_x_first = np.argmax(first_cluster[:, 0, 0])
    pts.append(first_cluster[min_x_first])
    pts.append(first_cluster[max_x_first])

    min_x_sec = np.argmin(second_cluster[:, 0, 0])
    max_x_sec = np.argmax(second_cluster[:, 0, 0])
    pts.append(second_cluster[min_x_sec])
    pts.append(second_cluster[max_x_sec])
    return pts


def draw_blobs(img, pts):
    detected = cv2.circle(img, (pts[0][0, 0], pts[0][0, 1]), 5, (255, 0, 0), -1)
    detected = cv2.circle(detected, (pts[1][0, 0], pts[1][0, 1]), 5, (255, 0, 0), -1)
    detected = cv2.circle(detected, (pts[2][0, 0], pts[2][0, 1]), 5, (255, 0, 0), -1)
    detected = cv2.circle(detected, (pts[3][0, 0], pts[3][0, 1]), 5, (255, 0, 0), -1)
    return detected

def blob_detection_lane(img, lines, situation='start'):
    """Possible values for situation (type: str)
    start: for when we only see the 5 meters line
    5_15: for when we see part of the 5m and the 15m mark
    15_25: ..
    25_35: ?
    35_45: ?
    end: ?"""
    lanes_side = select_outside_lines(lines)

    # drawing the outside line
    img = draw_lines_on_image(img, lanes_side)

    # blob detection
    keypoints = blop_detection_on_red(img)

    if situation == 'start' or situation == '45_50':
        try:
            pts = blob_selection_start(keypoints)
        except IndexError:
            # print("not enough clusters")
            return img, []
    else:
        try:
            pts = blob_selection_5_15(keypoints)
        except IndexError:
            # print("not enough clusters")
            return img, []
    # drawing
    if len(pts) >= 4:
        detected = draw_blobs(img, pts)
        return detected, pts
    else:
        return img, []

if __name__ == "__main__":
    # Load the image
    path = "./media/"
    file_name = "3lanes_1_entireimg.jpg"
    img = cv2.imread(path + file_name)

    # # for 3lane
    # lines_lanes = np.array([[[472.66666, 1.4486233]], [[394.33334, 1.4486233]], [[323, 1.4486233]]])
    # lanes_side = select_outside_lines(lines_lanes)

    # # for 3lane_lf
    # lines_lanes = np.array([[[260.5, 1.5358897]], [[418.25, 1.5358897]], [[334., 1.5358897]]])
    # lanes_side = select_outside_lines(lines_lanes)

    #for 3lanes_1
    lines_lanes = np.array([[[477., 1.3962634]], [[415., 1.3962634]], [[542., 1.3962634]]])
    lanes_side = select_outside_lines(lines_lanes)

    # # for 3lane_90
    # lines_lanes = np.array([[[391. , 1.4486233]], [[468., 1.4486233]], [[322.5 ,  1.4486233]]])
    # lanes_side = select_outside_lines(lines_lanes)

    # # 3 lanes_91
    # lines_lanes = np.array([[[474.33334, 1.4389269]], [[393.16666, 1.4471687]], [[318.33334, 1.4563804]]])
    # lanes_side = select_outside_lines(lines_lanes)

    # # for 3lanes_80
    # lines_lanes = np.array([[[322.7143, 1.4461299]], [[469.75, 1.4333516]], [[393.5, 1.4384421]]])
    # lanes_side = select_outside_lines(lines_lanes)

    # drawing the outside line
    img = draw_lines_on_image(img, lanes_side)
    # print_image(img, title='outside lines')
    #
    # blob detection
    keypoints = blop_detection_on_red(img)
    pts = blob_selection_start(keypoints)

    # pts = blop_selection_start(keypoints)
    # print(pts)

    #drawing
    detected = draw_blobs(img, pts)
    # print_image(detected, title="detected blobs")
    print(pts)

    # potential problems: bad detection becausedonc have the entire 5 meters
    # too many red zones on the image = too many clusters
    # change the definition of the image, need to change the hyperparameters of the clustering and of the selection

