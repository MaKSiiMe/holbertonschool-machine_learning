#!/usr/bin/env python3
"""5. Preprocess images"""
import tensorflow.keras as K
import numpy as np
import os
import cv2


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

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            box_xy = output[..., :2]
            box_wh = output[..., 2:4]

            col = np.arange(grid_width).reshape(1, grid_width, 1)
            col = np.tile(col, [grid_height, 1, anchor_boxes])
            row = np.arange(grid_height).reshape(grid_height, 1, 1)
            row = np.tile(row, [1, grid_width, anchor_boxes])

            sig_x = 1 / (1 + np.exp(-box_xy[..., 0]))
            box_xy[..., 0] = (sig_x + col) / grid_width
            sig_y = 1 / (1 + np.exp(-box_xy[..., 1]))
            box_xy[..., 1] = (sig_y + row) / grid_height

            anchor_wh = self.anchors[i]
            anchor_wh = anchor_wh.reshape(1, 1, anchor_boxes, 2)
            box_wh = anchor_wh * np.exp(box_wh)

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]
            box_wh[..., 0] /= input_width
            box_wh[..., 1] /= input_height

            box_x1 = (box_xy[..., 0] - box_wh[..., 0] / 2) * image_width
            box_y1 = (box_xy[..., 1] - box_wh[..., 1] / 2) * image_height
            box_x2 = (box_xy[..., 0] + box_wh[..., 0] / 2) * image_width
            box_y2 = (box_xy[..., 1] + box_wh[..., 1] / 2) * image_height

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = box_x1
            box[..., 1] = box_y1
            box[..., 2] = box_x2
            box[..., 3] = box_y2

            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on class threshold
        """
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

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply Non-Maximum Suppression
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idx = np.where(box_classes == cls)

            class_boxes = filtered_boxes[idx]
            class_scores = box_scores[idx]

            sorted_idx = np.argsort(class_scores)[::-1]
            class_boxes = class_boxes[sorted_idx]
            class_scores = class_scores[sorted_idx]

            keep_boxes = []
            keep_scores = []

            while len(class_boxes) > 0:
                keep_boxes.append(class_boxes[0])
                keep_scores.append(class_scores[0])

                if len(class_boxes) == 1:
                    break

                x1 = np.maximum(class_boxes[0, 0], class_boxes[1:, 0])
                y1 = np.maximum(class_boxes[0, 1], class_boxes[1:, 1])
                x2 = np.minimum(class_boxes[0, 2], class_boxes[1:, 2])
                y2 = np.minimum(class_boxes[0, 3], class_boxes[1:, 3])

                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                box1_area = ((class_boxes[0, 2] - class_boxes[0, 0]) *
                             (class_boxes[0, 3] - class_boxes[0, 1]))
                box2_area = ((class_boxes[1:, 2] - class_boxes[1:, 0]) *
                             (class_boxes[1:, 3] - class_boxes[1:, 1]))
                union = box1_area + box2_area - intersection

                iou = intersection / union

                keep_mask = iou < self.nms_t
                class_boxes = class_boxes[1:][keep_mask]
                class_scores = class_scores[1:][keep_mask]

            if len(keep_boxes) > 0:
                box_predictions.append(np.array(keep_boxes))
                predicted_box_classes.append(np.full(len(keep_boxes), cls))
                predicted_box_scores.append(np.array(keep_scores))

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_scores = np.concatenate(predicted_box_scores, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Load images from a folder
        """
        images = []
        image_paths = []

        for filename in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                image = cv2.imread(filepath)
                if image is not None:
                    images.append(image)
                    image_paths.append(filepath)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocess images for the Darknet model
        Args:
            images: list of images as numpy.ndarrays
        Returns:
            tuple of (pimages, image_shapes):
                pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                image_shapes: numpy.ndarray of shape (ni, 2)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_LINEAR)
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Display the image with all boundary boxes, class names, and box scores
        """
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Draw blue box with thickness 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Prepare text with class name and score
            class_name = self.class_names[int(box_classes[i])]
            score = box_scores[i]
            text = "{} {:.2f}".format(class_name, score)
            
            # Draw red text 5 pixels above the top left corner
            cv2.putText(image, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 
                       1, cv2.LINE_AA)
        
        # Display the image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        
        # If 's' key is pressed, save the image
        if key == ord('s'):
            # Create detections directory if it doesn't exist
            if not os.path.exists('detections'):
                os.makedirs('detections')
            
            # Save the image
            save_path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(save_path, image)
        
        # Close the window
        cv2.destroyAllWindows()
