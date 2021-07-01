import numpy as np
import cv2
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
    return keypoints, filtered

if __name__ == "__main__":
    # Load the image
    path = "./media/"
    file_name = "3lanes_1_entireimg.jpg"
    img = cv2.imread(path + file_name)
    original = img.copy()

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
    # cv2.imshow("Lanes", img)
    # cv2.waitKey(0)

    #
    # blob detection
    keypoints_scene, red = blop_detection_on_red(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints_scene, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Blobs", im_with_keypoints)
    # cv2.waitKey(0)

    object = cv2.imread("./media/halfspa_ry.jpg")
    keypoints_obj, _ = blop_detection_on_red(object)
    object_blob = cv2.drawKeypoints(object, keypoints_obj, np.array([]), (255, 0, 0),
                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Blobs obj", object_blob)
    # cv2.waitKey(0)

    detector = cv2.ORB_create()
    # keypoints_obj = detector.detect(img_object, None)
    keypoints_obj, descriptors_obj = detector.compute(red, keypoints_obj)
    # keypoints_scene = detector.detect(img_scene, None)
    keypoints_scene, descriptors_scene = detector.compute(object, keypoints_scene)
    print(len(keypoints_obj), len(keypoints_scene))

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher_create()
    matches = bf.match(descriptors_obj, descriptors_scene)
    good_matches = sorted(matches, key=lambda x: x.distance)

    img_matches = np.empty((max(object.shape[0], img.shape[0]), object.shape[1] + img.shape[1], 3),
                           dtype=np.uint8)
    cv2.drawMatches(object, keypoints_obj, img, keypoints_scene, good_matches, img_matches,
                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow("good_matches", img_matches)
    # cv2.waitKey(0)

    obj_cokey = np.array([[int(keypoint.pt[0]), int(keypoint.pt[1])] for keypoint in keypoints_obj])
    scene_cokey = np.array([[int(keypoint.pt[0]), int(keypoint.pt[1])] for keypoint in keypoints_scene])

    obj = []
    scene = []
    for i in range(len(obj_cokey)):
        for j in range(len(scene_cokey)):
            # -- Get the keypoints from the good matches
            obj.append(obj_cokey[i])
            scene.append(scene_cokey[j])
    obj = np.array(obj)
    scene = np.array(scene)

    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)
    print(H)

    dst = cv2.warpPerspective(original, H, (object.shape[0], object.shape[1]))
    cv2.namedWindow('apply homography', cv2.WINDOW_NORMAL)
    cv2.imshow('apply homography', dst)
    cv2.waitKey(0)

    cv2.imwrite("./media/red_1.jpg", red)