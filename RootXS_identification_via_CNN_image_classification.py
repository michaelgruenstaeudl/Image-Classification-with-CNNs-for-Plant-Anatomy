#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN for classifying dicot vs monocot root cross sections
Author: Michael Gruenstaeudl, PhD | Email: m_gruenstaeudl@fhsu.edu
"""

from __future__ import annotations

import os
from glob import glob
from typing import Tuple, List

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

################################################################################
# GLOBAL VARIABLES
IMG_W, IMG_H = 150, 150
BATCH_SIZE = 10
EPOCHS = 10
TRAIN_DIR = "data/train/"
TEST_DIR = "data/test/"
VAL_SPLIT = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)



################################################################################
# LOCATE THE IMAGE DATA ON DISK, RESIZE IMAGES, AND RUN CODE

def load_test_images(paths: List[str], size: Tuple[int, int]) -> torch.Tensor:
''' This function does the following: It
        1. reads images using OpenCV (cv2.imread),
        2. converts the color definition of all images from BGR (how OpenCV reads images) to RGB standard (cv2.COLOR_BGR2RGB),
        3. resizes all images to (IMG_W, IMG_H) (i.e., standardize spatial dimensions of all images)
        4. scales images (division by 255) so that optimization via neural network occurs quicker
        5. standardizes how image data is laid out in memory (i.e., standardize order of dimensions of image tensor)
        6. stacks all images into a single tensor of shape (N, 3, H, W)
'''
    w, h = size
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # Reads in BGR color definition
        if img is None:
            raise RuntimeError(f"Unable to read image: {p}")
        # The following lines of code standardize the images regarding size,
        # color encoding, etc.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts to RGB color definition
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  # [0, 1]
        img = np.transpose(img, (2, 0, 1))    # HWC -> CHW (PyTorch expects channels (C) first)
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"No test images present in: {TEST_DIR}")
    # Returning images as a single tensor of shape (N, 3, H, W)
    return torch.from_numpy(np.stack(imgs, axis=0))

# STEP 1. Load the test images
test_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))
x_test = load_test_images(test_paths, (IMG_W, IMG_H)).to(DEVICE)

# STEP 2. Switch CNN to evaluation mode
model.eval()

# STEP 3. Run CNN with disabled gradients (for better speed and memory efficiency)
with torch.no_grad():
    # 3.1. Infer the logits for the model (i.e., the unnormalized output of the final linear layer)
    logits = model(x_test)
    # 3.2. Convert logits to probabilities via sigmoid function
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    # 3.3. Infer predicted classes with probabilities at 0.5 or larger
    preds = (probs >= 0.5).astype(int)

print("Predicted probabilities:", probs)
print("Predicted classes (0 or 1):", preds)
print("Training class_to_idx:", full_ds.class_to_idx) # Print training labels to mapping (use folder name as index names)

################################################################################
# EXTENDED USAGE

# If labels for test images are available, model accuracy can be calculated:
# y_test = np.array([...], dtype=np.int64)
# test_acc = (preds == y_test).mean()
# print("Test accuracy:", test_acc)
