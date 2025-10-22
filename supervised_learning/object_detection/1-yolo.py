#!/usr/bin/env python3
"""1. Process Outputs"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract box confidence (sigmoid of 5th value)
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            # Extract class probabilities (sigmoid of class values)
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            # Process boundary boxes
            box_xy = output[..., :2]
            box_wh = output[..., 2:4]

            # Create grid coordinates
            col = np.arange(grid_width).reshape(1, grid_width, 1)
            col = np.tile(col, [grid_height, 1, anchor_boxes])
            row = np.arange(grid_height).reshape(grid_height, 1, 1)
            row = np.tile(row, [1, grid_width, anchor_boxes])

            # Apply YOLO formulas
            # bx = sigmoid(tx) + cx
            # by = sigmoid(ty) + cy
            sig_x = 1 / (1 + np.exp(-box_xy[..., 0]))
            box_xy[..., 0] = (sig_x + col) / grid_width
            sig_y = 1 / (1 + np.exp(-box_xy[..., 1]))
            box_xy[..., 1] = (sig_y + row) / grid_height

            # bw = pw * e^(tw)
            # bh = ph * e^(th)
            anchor_wh = self.anchors[i]
            anchor_wh = anchor_wh.reshape(1, 1, anchor_boxes, 2)
            box_wh = anchor_wh * np.exp(box_wh)

            # Normalize by model input size
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            box_wh[..., 0] /= input_width
            box_wh[..., 1] /= input_height

            # Convert to corner coordinates (x1, y1, x2, y2)
            box_x1 = (box_xy[..., 0] - box_wh[..., 0] / 2) * image_width
            box_y1 = (box_xy[..., 1] - box_wh[..., 1] / 2) * image_height
            box_x2 = (box_xy[..., 0] + box_wh[..., 0] / 2) * image_width
            box_y2 = (box_xy[..., 1] + box_wh[..., 1] / 2) * image_height

            # Stack coordinates
            box = np.zeros(output[..., :4].shape)
            box[..., 0] = box_x1
            box[..., 1] = box_y1
            box[..., 2] = box_x2
            box[..., 3] = box_y2

            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)
