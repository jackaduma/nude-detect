#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-06-08 14:49:43
LastEditTime: 2022-06-08 15:09:51
LastEditors: Kun
Description: 
FilePath: /my_open_projects/nude-detect/main.py
'''

import os
import argparse

from classifier import Classifier
from config import cls_model_path


def clf_images():
    m = Classifier(cls_model_path)
    unsafe_img = "./data/image/nude/0B6FE142-67A9-4451-B977-6E22C0EC12D7.jpg"
    safe_img = "./data/image/normal/030u5wa70z6z.jpg"
    img_path_1 = "./data/image/nude/0D16FBAD-655B-440C-A7E1-32D20408DF40.jpg"  # real

    images_preds = m.classify([unsafe_img, safe_img, img_path_1])

    print("# preds -------")
    for k, v in images_preds.items():
        print(k, v)


def clf_video():
    m = Classifier(cls_model_path)
    video_path = "./data/video/123.mp4"

    result = m.classify_video(video_path)
    metadata_info = result["metadata"]
    preds_info = result["preds"]

    print("# metadata -------")
    for k, v in metadata_info.items():
        print(k, v)

    print("# preds -------")
    for k, v in preds_info.items():
        print(k, v)


def clf():
    m = Classifier(cls_model_path)

    while 1:
        print(
            "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
        )
        images = input().split("||")
        images = [image.strip() for image in images]
        print(m.classify(images), "\n")


if __name__ == "__main__":
    # clf_images()

    clf_video()
