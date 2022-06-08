#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-30 23:28:03
LastEditTime: 2022-06-08 17:58:48
LastEditors: Kun
Description: 
FilePath: /my_open_projects/nude-detect/detector.py
'''

import os
import keras
import pydload
from keras_retinanet import models
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.models.retinanet import retinanet_bbox, retinanet
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

from utils.video_utils import get_interest_frames_from_video

import cv2
import numpy as np

import logging

from PIL import Image as pil_image

from progressbar import progressbar


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    if isinstance(path, str):
        image = np.ascontiguousarray(pil_image.open(path).convert("RGB"))
    else:
        path = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(pil_image.fromarray(path))

    print(image.shape)
    final_img = image[:, :, ::-1]
    print(final_img.shape)
    return final_img


def dummy(x):
    return x


class Detector(object):
    detection_model = None
    classes = None

    def __init__(self, checkpoint_path, classes_path, model_name="default"):
        """
            model = Detector()
        """

        self.classes = [
            c.strip() for c in open(classes_path).readlines() if c.strip()
        ]
        print("# classes: ----")
        print(self.classes)
        self.detection_model = models.load_model(
            checkpoint_path, backbone_name="resnet50"
        )
        # self.detection_model = retinanet_bbox(
        #     inputs=[3, None, None], backbone_layers=["C3", "C4", "C5"],
        #     num_classes=len(self.classes)
        # )
        # self.detection_model.load_weights(checkpoint_path)

    def detect_video(self, video_path, min_prob=0.6, batch_size=2, show_progress=True):
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(
            video_path
        )
        logging.debug(
            f"VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}"
        )
        frames = [read_image_bgr(frame) for frame in frames]
        frames = [preprocess_image(frame) for frame in frames]
        frames = [resize_image(frame) for frame in frames]
        scale = frames[0][1]
        frames = [frame[0] for frame in frames]
        all_results = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": {},
        }

        progress_func = progressbar

        if not show_progress:
            progress_func = dummy

        for _ in progress_func(range(int(len(frames) / batch_size) + 1)):
            batch = frames[:batch_size]
            batch_indices = frame_indices[:batch_size]
            frames = frames[batch_size:]
            frame_indices = frame_indices[batch_size:]
            if batch_indices:
                boxes, scores, labels = self.detection_model.predict_on_batch(
                    np.asarray(batch)
                )
                boxes /= scale
                for frame_index, frame_boxes, frame_scores, frame_labels in zip(
                    frame_indices, boxes, scores, labels
                ):
                    if frame_index not in all_results["preds"]:
                        all_results["preds"][frame_index] = []

                    for box, score, label in zip(
                        frame_boxes, frame_scores, frame_labels
                    ):
                        if score < min_prob:
                            continue
                        box = box.astype(int).tolist()
                        label = self.classes[label]

                        all_results["preds"][frame_index].append(
                            {"box": box, "score": score, "label": label}
                        )

        return all_results

    def detect(self, img_path, min_prob=0.6):
        image = read_image_bgr(img_path)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = self.detection_model.predict_on_batch(
            np.expand_dims(image, axis=0)
        )
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = self.classes[label]
            processed_boxes.append(
                {"box": box, "score": score, "label": label})

        return processed_boxes

    def censor(self, img_path, out_path=None, visualize=False, parts_to_blur=[]):
        if not out_path and not visualize:
            print(
                "No out_path passed and visualize is set to false. There is no point in running this function then."
            )
            return

        image = cv2.imread(img_path)
        boxes = self.detect(img_path)

        if parts_to_blur:
            boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
        else:
            boxes = [i["box"] for i in boxes]

        for box in boxes:
            part = image[box[1]: box[3], box[0]: box[2]]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2],
                                          box[3]), (0, 0, 0), cv2.FILLED
            )

        if visualize:
            cv2.imshow("Blurred image", image)
            cv2.waitKey(0)

        if out_path:
            cv2.imwrite(out_path, image)


def label2cn(label):
    d = {
        'EXPOSED_BELLY': "裸露-腹部",
        'EXPOSED_BUTTOCKS': "裸露-臀部",
        'EXPOSED_BREAST_F': "裸露-胸部-女",
        'EXPOSED_GENITALIA_F': "裸露-生殖器-女",
        'EXPOSED_GENITALIA_M': "裸露-生殖器-男",
        'EXPOSED_BREAST_M': "裸露-胸部-男",
    }

    return d[label]


if __name__ == "__main__":
    from config import base_detect_model_path, base_detect_class_path, default_detect_model_path, default_detect_class_path, detect_model_path
    m = Detector(base_detect_model_path, base_detect_class_path)
    # m = Detector(default_detect_model_path, default_detect_class_path)

    img_path_1 = "./data/image/nude/0D16FBAD-655B-440C-A7E1-32D20408DF40.jpg"  # real
    img_path_2 = "./data/image/nude/0B6FE142-67A9-4451-B977-6E22C0EC12D7.jpg"  # cartoon

    print("# detect: -----------")
    results = m.detect(img_path_1)
    for res in results:
        box = res["box"]
        score = res["score"]
        label = res["label"]
        label_cn = label2cn(label)
        print(label, label_cn, score, box)

    marked_file_path = "./output/res1.jpg"
    print("# mark: ---------- {}".format(marked_file_path))
    m.censor(img_path_1, out_path=marked_file_path)

    
    print("#detect: -----------")
    results = m.detect(img_path_2)
    for res in results:
        box = res["box"]
        score = res["score"]
        label = res["label"]
        label_cn = label2cn(label)
        print(label, label_cn, score, box)

    marked_file_path = "./output/res2.jpg"
    print("# mark: -----------  {}".format(marked_file_path))
    print(m.censor(img_path_2, out_path=marked_file_path))
