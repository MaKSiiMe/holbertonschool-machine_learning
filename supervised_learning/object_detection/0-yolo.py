#!/usr/bin/env python3
"""0. Initialize Yolo"""
from tensorflow import keras as K


class Yolo:
    """
    Class that uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize the Yolo class
        """
        # Load the Darknet Keras model
        self.model = K.models.load_model(model_path)

        # Load class names from file
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        # Set thresholds
        self.class_t = class_t
        self.nms_t = nms_t

        # Set anchor boxes
        self.anchors = anchors
