"""
This class allows the user to put the information about some points.
"""
from PyQt5.QtWidgets import QHBoxLayout, QGridLayout
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from src.d0_utils.point_selection.information_points.edit_point.auto_edit_point import EditPoint
from src.d0_utils.point_selection.information_points.text_point.text_point import TextPoint


INSTRUCTIONS = "INSTRUCTIONS : \n \n"
INSTRUCTIONS += "   Select a point : left click. \n \n"
INSTRUCTIONS += "   Zoom in : keep left click, move and release. \n \n"
INSTRUCTIONS += "   Zoom out : click anywhere. \n \n"
INSTRUCTIONS += "   Withdraw a point : press 'w'. \n \n"
INSTRUCTIONS += "   Quit : press space bar."


class CalibrationPoints(QGridLayout):
    """
    QGridLayout class that displays QTextEdit objects.
    """
    def __init__(self, size, colors, points):
        """
        Construct the objects to manage the QTextEdit objects.

        Args:
            size (QSize): the size of the QGridLayout.

            colors (list of Qt.Color): the colors of the points.

            points (array, shape = (len(colors), 2)): the points on which the information
                will be registered.
        """
        super().__init__()
        self.colors = colors
        self.nb_points = len(colors)
        self.size = size
        smallest_height = size.height() / (2 * self.nb_points + 4)  # + 4 for the instructions
        self.size_text = QSize(self.size.width(), smallest_height)
        self.size_edit_text = QSize(self.size.width() / 4, smallest_height)
        self.size_edit = QSize(self.size.width() / 4, smallest_height / 2)
        self.size_instructions = QSize(self.size.width(), 4 * smallest_height)
        self.points = points

        self.set_raw_labels()

    def set_raw_labels(self):
        """
        Fill the QGridLayout with QLabel and QTextEdit.
        """
        for index_color in range(self.nb_points):
            color = self.colors[index_color]
            point_layout = QHBoxLayout()

            # color_point = TextPoint("Point {}".format(index_color + 1), self.size_text)

            # edit_meter = EditPoint(self.size_edit, color, self.points, index_color, 0)
            # point_layout.addWidget(edit_meter)

            # text_meter = TextPoint("meters", self.size_edit_text)
            # point_layout.addWidget(text_meter)

            text_class = TextPoint("Point {}".format(index_color + 1), self.size_edit_text)
            point_layout.addWidget(text_class)

            edit_line = EditPoint(self.size_edit, color, self.points, index_color)
            point_layout.addWidget(edit_line)

            # self.addWidget(color_point, 2 * index_color, 0)
            self.addLayout(point_layout, index_color, 0)

        instructions = TextPoint(INSTRUCTIONS, self.size_instructions)
        instructions.setAlignment(Qt.AlignLeft)
        self.addWidget(instructions, 2 * self.nb_points + 1, 0)
