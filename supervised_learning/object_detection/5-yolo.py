#!/usr/bin/env python3
"""Yolo v3 algorithm for object detection."""

import numpy as np
from tensorflow import keras as K
import os
import cv2


class Yolo:
    """Class defining yolo v3 algorithm for object detection."""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor for Yolo v3 algorithm."""
        self.model = K.models.load_model(model_path)

        f = open(classes_path, 'r')
        self.class_names = f.read().split("\n")[:-1]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Processes the outputs.
        Returns the boundary boxes, box confidence's
        and class probabilities for each output."""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box_xy, box_wh = output[..., :2], output[..., 2:4]
            box_confidence, class_prob = output[..., 4:5], output[..., 5:]

            box_xy = 1 / (1 + np.exp(-box_xy))
            box_confidence = 1 / (1 + np.exp(-box_confidence))
            class_prob = 1 / (1 + np.exp(-class_prob))

            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width), np.arange(grid_height)
            )
            grid = np.stack(
                (grid_x, grid_y), axis=-1
            ).reshape((grid_height, grid_width, 1, 2))

            box_xy = (box_xy + grid) / [grid_width, grid_height]
            anchors = self.anchors[i].reshape((1, 1, anchor_boxes, 2))
            box_wh = np.exp(box_wh) * anchors
            box_x1y1 = box_xy - (box_wh / 2)
            box_x2y2 = box_xy + (box_wh / 2)
            box = np.concatenate([box_x1y1, box_x2y2], axis=-1)

            box[..., 0] *= image_width
            box[..., 1] *= image_height
            box[..., 2] *= image_width
            box[..., 3] *= image_height

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters bounding boxes by confidence score, class probabilities"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)
            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Selects bounding boxes based on non-maximum suppression."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_labels = np.unique(box_classes)

        for label in unique_labels:
            mask = box_classes == label
            class_boxes = filtered_boxes[mask]
            class_scores = box_scores[mask]

            indices = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[indices]
            class_scores = class_scores[indices]

            while len(class_boxes) > 0:
                highest_conf_box = class_boxes[0]
                highest_conf_score = class_scores[0]

                box_predictions.append(highest_conf_box)
                predicted_box_classes.append(label)
                predicted_box_scores.append(highest_conf_score)

                x1 = np.maximum(highest_conf_box[0], class_boxes[1:, 0])
                y1 = np.maximum(highest_conf_box[1], class_boxes[1:, 1])
                x2 = np.minimum(highest_conf_box[2], class_boxes[1:, 2])
                y2 = np.minimum(highest_conf_box[3], class_boxes[1:, 3])

                intersection_x = np.maximum(0, x2 - x1)
                intersection_y = np.maximum(0, y2 - y1)
                intersection_area = intersection_x * intersection_y

                box_w = (class_boxes[:, 2] - class_boxes[:, 0])
                box_h = (class_boxes[:, 3] - class_boxes[:, 1])
                box_area = box_w * box_h

                highest_conf_w = (highest_conf_box[2] - highest_conf_box[0])
                highest_conf_h = (highest_conf_box[3] - highest_conf_box[1])
                highest_conf_area = highest_conf_w * highest_conf_h
                union = box_area[1:] + highest_conf_area - intersection_area

                iou = intersection_area / union

                iou_mask = iou < self.nms_t
                class_boxes = class_boxes[1:][iou_mask]
                class_scores = class_scores[1:][iou_mask]
        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )

    @staticmethod
    def load_images(folder_path):
        """Loads images from folder.
        Returns list of images and list of images paths."""
        images = []
        images_paths = []
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                images_paths.append(image_path)
        return images, images_paths

    def preprocess_images(self, images):
        """Resizes images with inter-cubic interpolation
        and normalizes pixel values."""
        image_shapes = []
        pimages = []

        input_w, input_h = self.model.input.shape[1:3]

        for image in images:
            image_height, image_width = image.shape[:2]
            image_shapes.append((image_height, image_width))
            resized_images = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            norm_image = resized_images / 255.0
            pimages.append(norm_image)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
