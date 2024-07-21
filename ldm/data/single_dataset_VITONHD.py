from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from numpy import asarray

import random
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import os

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

class SingleImageTryOnDataset:
    def __init__(self, preprocessed_set_path):
        self.preprocessed_set_path = preprocessed_set_path
        self.kernel = np.ones((1, 1), np.uint8)
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.boundingbox_as_inpainting_mask_rate = 0.4

    def process(self):
        # Load images
        source_img = Image.open(os.path.join(self.preprocessed_set_path, "image.png")).convert("RGB")
        source_img = source_img.resize((384, 512), Image.BILINEAR)
        image_tensor = get_tensor()(source_img)

        segment_map = Image.open(os.path.join(self.preprocessed_set_path, "imageparse.png"))
        segment_map = segment_map.resize((384, 512), Image.NEAREST)
        parse_array = np.array(segment_map)

        ref_img = Image.open(os.path.join(self.preprocessed_set_path, "cloth.png")).convert("RGB")
        ref_img = ref_img.resize((384, 512), Image.BILINEAR)
        ref_img_tensor = get_tensor()(ref_img)

        pose_img = Image.open(os.path.join(self.preprocessed_set_path, "openpose.png")).convert("RGB")
        pose_img = pose_img.resize((384, 512), Image.BILINEAR)
        pose_tensor = get_tensor()(pose_img)

        densepose_img = Image.open(os.path.join(self.preprocessed_set_path, "densepose.png")).convert("RGB")
        densepose_img = densepose_img.resize((384, 512), Image.BILINEAR)
        densepose_tensor = get_tensor()(densepose_img)

        # Process garment mask
        garment_mask = ((parse_array == 5) | (parse_array == 7)).astype(np.float32)
        garment_mask_with_arms = ((parse_array == 5) | (parse_array == 7) | (parse_array == 14) | (parse_array == 15)).astype(np.float32)

        # Generate inpainting mask
        garment_mask = 1 - garment_mask
        garment_mask[garment_mask < 0.5] = 0
        garment_mask[garment_mask >= 0.5] = 1
        garment_mask_resized = cv2.resize(garment_mask, (384, 512), interpolation=cv2.INTER_NEAREST)

        # Contour processing (similar to original code)
        contours, _ = cv2.findContours(((1 - garment_mask_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            max_contour = max(contours, key = cv2.contourArea)
            epsilon = 0.003 * cv2.arcLength(max_contour, closed=True)  # Using a fixed epsilon for simplicity
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, closed=True)
            randomness = np.random.randint(-90, 90, approx_contour.shape)  # Using fixed randomness range for simplicity
            approx_contour = approx_contour + randomness

            zero_mask = np.zeros((512, 384))
            contours = [approx_contour]

            cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((100,100),np.uint8)  # Using fixed kernel size for simplicity
            garment_mask_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
            garment_mask_inpainting = garment_mask_inpainting.astype(np.float32) / 255.0
            garment_mask_inpainting[garment_mask_inpainting < 0.5] = 0
            garment_mask_inpainting[garment_mask_inpainting >= 0.5] = 1
            garment_mask_inpainting = garment_mask_resized * (1 - garment_mask_inpainting)
        else:
            garment_mask_inpainting = np.zeros((512, 384))

        # Generate GT mask and inpainting mask
        garment_mask_GT = cv2.erode(garment_mask_resized, self.kernel_dilate, iterations=3)[None]
        garment_mask_inpainting = cv2.erode(garment_mask_inpainting, self.kernel_dilate, iterations=5)[None]

        garment_mask_GT_tensor = torch.from_numpy(garment_mask_GT)
        garment_mask_inpainting_tensor = torch.from_numpy(garment_mask_inpainting)

        # Generate bounding box
        garment_mask_with_arms = 1 - garment_mask_with_arms
        garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
        garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
        garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384, 512), interpolation=cv2.INTER_NEAREST)
        garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]

        _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
        if x.size > 0 and y.size > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
            boundingbox[:, y_min:y_max, x_min:x_max] = 0
        else:
            boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

        boundingbox_tensor = torch.from_numpy(boundingbox)

        # Prepare inpainting mask
        inpainting_mask_tensor = boundingbox_tensor

        # Prepare output tensors
        inpaint_image = image_tensor * inpainting_mask_tensor
        GT_image_combined = torch.cat((image_tensor, ref_img_tensor), dim=2)
        GT_mask_combined = torch.cat((garment_mask_GT_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)
        inpaint_image_combined = torch.cat((inpaint_image, ref_img_tensor), dim=2)
        inpainting_mask_combined = torch.cat((inpainting_mask_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)
        pose_combined = torch.cat((pose_tensor, ref_img_tensor), dim=2)
        densepose_combined = torch.cat((densepose_tensor, ref_img_tensor), dim=2)

        # Prepare output dictionary
        return {
            "image_name": ["single_image_set"],  # Make this a list
            "GT_image": GT_image_combined.unsqueeze(0),
            "GT_mask": GT_mask_combined.unsqueeze(0),
            "inpaint_image": inpaint_image_combined.unsqueeze(0),
            "inpaint_mask": inpainting_mask_combined.unsqueeze(0),
            "posemap": pose_combined.unsqueeze(0),
            "densepose": densepose_combined.unsqueeze(0),
            "ref_list": [ref_img_tensor.unsqueeze(0)],
        } 

# Usage example
# dataset = SingleImageTryOnDataset("preprocessed/test_input")
# result = dataset.process()
