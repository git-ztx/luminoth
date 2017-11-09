import abc

from luminoth.tools.dataset.readers import BaseReader


class ObjectDetectionReader(BaseReader):
    """Reads data suitable for object detection.

    Object detections needs:
        - images
        - gt_boxes with labels
    """
    def __init__(self):
        super(ObjectDetectionReader, self).__init__()

    @abc.abstractproperty
    def classes(self):
        """Returns a list of class names available in dataset.
        """

    def set_classes(self, classes):
        self._classes = classes
