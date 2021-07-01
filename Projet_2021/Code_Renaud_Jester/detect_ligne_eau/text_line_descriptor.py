import cv2
import numpy as np
from pylsd import lsd

if __name__ == "__main__":
    path = "./media/"
    file_name = "brasse_50_fframe.png"
    img = cv2.imread(path + file_name)

    # # descriptor = cv2.line_descriptor_LSDDetector()
    # retval = cv2.line_descriptor.LSDDetector_createLSDDetector()
    # key_lines = np.array([])
    # key_lines = retval.detect(gray, key_lines, 10, 5)
    # outImage = cv2.line_descriptor.drawKeylines(img, key_lines)

    # descriptor using pylsd-nova
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = lsd(gray, scale=0.5)
    for i in range(lines.shape[0]):
        pt1 = (int(lines[i, 0]), int(lines[i, 1]))
        pt2 = (int(lines[i, 2]), int(lines[i, 3]))
        width = lines[i, 4]
        cv2.line(img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
    cv2.imshow('Test', img)
    cv2.waitKey(0)


