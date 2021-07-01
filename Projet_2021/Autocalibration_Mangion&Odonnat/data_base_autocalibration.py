"""
The file allows the user to select points in an image,
to tell the real position of these points.
"""

from pathlib import Path
import numpy as np
import cv2

# For the application
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

# To plot an image and select a point in it
from src.d0_utils.point_selection.image_selection.image_selection import ImageSelection

# To plot the information about the points
from src.d0_utils.point_selection.information_points.autocalibration_points import CalibrationPoints

# To set the main widget
from src.d0_utils.point_selection.main_widget.main_widget import MainWidget

# To get the instructions for calibration
from src.d0_utils.point_selection.instructions.instructions import instructions_calibration

import skimage.io as io

def array_to_qpixmap(image):
    """
    Transform an array in a QPixmap.

    Args:
        image (array, 2 dimensions, bgr format): the image.

    Returns:
        (QPixmap, brg format): the QPixmap.
    """
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    # If the format is not good : put Format_BGR888
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

    return QPixmap.fromImage(qimage)


def calibration_selection(image):
    """
    Displays an image on which the user can select points and tell their positions.

    Args:
        image (array, 2 dimensions): the image on which the points will be selected.

    Returns:
        points_image (array, shape = (4, 2)): the select points.

        points_real (array, shape = (4, 2)): the given points.

    Interaction events :
        - if the user click a point is selected
        - if the user click and drag, it zooms
        - if the user press the escape button, it erases the last point
        - if the user press the space bar, the application is closed.
    """
    # Set the points
    colors = [Qt.red]*14
    points_image = np.ones((len(colors), 2), dtype=np.float32) * -2
    points_real = np.ones((len(colors)), dtype=np.float32) * -2

    # Set application, window and layout
    app = QApplication([])
    instructions_calibration()
    window = MainWidget()
    layout = QHBoxLayout()

    # Get the sizes
    screen_size = QDesktopWidget().screenGeometry()
    screen_ration = 4 / 5
    image_size = QSize(screen_size.width() * screen_ration - 50, screen_size.height() - 150)
    # image_size = Qsize(image.shape[0], image.shape[1])
    point_size = QSize(screen_size.width() * (1 - screen_ration) - 50, screen_size.height() - 150)

    # Set the image selection and the editable text
    pix_map = array_to_qpixmap(image)
    image_selection = ImageSelection(pix_map, image_size, points_image, colors, skip=True)
    information_points = CalibrationPoints(point_size, colors, points_real)

    # Add widgets to layout
    layout.addWidget(image_selection)
    layout.addLayout(information_points)

    # Add layout to window and show the window
    window.setLayout(layout)
    window.showMaximized()
    app.exec_()

    return points_image, points_real

def float_to_uint8(img, multiply_channel=False):
  if multiply_channel:
    return (255*np.stack((img,)*3, axis=-1)).astype(np.uint8)
  return (255*img).astype(np.uint8)


def print_heatmap(heatmap):
  display = cv2.GaussianBlur(float_to_uint8(heatmap),(21,21),5.0)
  io.imshow(display)
  io.show()


if __name__ == "__main__":
    # Get the image
    k = 1
    # test_range = [1, 2, 3, 4, 5, 6, 8, 27, 46, 66, 90, 95, 104, 105, 108, 109, 198, 199]

    image_name = "X" + str(k) + ".jpg"
    path = "E:/IMIP2/image_db/"
    ROOT_IMAGE = Path(path + image_name)
    IMAGE = cv2.imread(str(ROOT_IMAGE))

    # initialize label
    label = np.zeros((8, 512, 512), float)

    # graphic interface for labelling
    points_image, n_class = calibration_selection(IMAGE)

    # fill label
    for i, p in enumerate(points_image):
        if n_class[i]>=0:
            label[int(n_class[i]), int(p[1]), int(p[0])] = 1.

    # np.save("E:/IMIP2/label_db/label" + str(k) + ".npy", label)



    # # Debug
    # check = np.load("E:/IMIP2/label_db/label" + str(k) + ".npy")
    # for i in range(8):
    #     cv2.imshow("check" + str(k)+ "." + str(i), check[i])
    #     cv2.waitKey()
