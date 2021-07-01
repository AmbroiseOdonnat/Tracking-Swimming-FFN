"""
Allows the user to put the information about the points he/she have selected.
"""
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt


class EditPoint(QTextEdit):
    """
    QTextEdit class.
    """
    def __init__(self, size, color, points, index_point):
        """
        Construct the QTextEdit class.

        Args:
            size (QSize): the size of the QTextEdit object.

            color (QT.Color): the color of the text.

            points (array, shape = (_, 2)): the list of point that gather the information
                on the selected points.

            index_point (int): this object will ve link to the selected point
                with the index "index_point".

            meter_or_line (int, 0 or 1): if 1 indicates of the information is about
                the vertical coordinate.
        """
        super().__init__()
        self.setTabChangesFocus(True)
        self.setFixedSize(size)
        self.color = color
        self.setTextColor(self.color)
        self.points = points
        self.index_point = index_point

    def keyReleaseEvent(self, event):
        """
        If escape is pressed, the last selected point is erased.
        If space bar is pressed, the entire widget is closed.
        """
        self.setTextColor(self.color)

        if event.key() == Qt.Key_W:
            self.parentWidget().erase_point()

        if event.key() == Qt.Key_Space:
            self.parentWidget().close()

    def focusOutEvent(self, event):
        """
        Register the information given by the user when it is closed.
        """
        # If something has been written
        if self.toPlainText() != "":
            # If a space has been written
            if " " in self.toPlainText():
                self.textCursor().deletePreviousChar()
            # If the written text is a float
            if self.toPlainText().isnumeric():
                self.points[self.index_point] = float(self.toPlainText())
            # We erase the written text
            else:
                self.clear()
