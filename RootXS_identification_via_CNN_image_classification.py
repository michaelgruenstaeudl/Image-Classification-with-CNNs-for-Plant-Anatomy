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
# ACTUALLY TRAIN THE DATA

# STEP 1. Instantiate the model and move it to DEVICE (GPU if available)
model = RootCNN().to(DEVICE)

# STEP 2. Specify the loss function to measure how well the predicted probabilities match the binary targets
criterion = nn.BCEWithLogitsLoss()  # better than sigmoid activation and binary cross-entropy loss function in Keras

# STEP 3. Specify the output optimization using RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)  # roughly analogous to Keras "rmsprop"

# STEP 4 (ML OCCRUS HERE). Iterating over all samples (i.e. epoch) to calculate loss functionn and perform optimization
# Each epoch runs one training pass over train_loader and one evaluation pass over val_loader
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion)
    va_loss, va_acc = run_epoch(model, val_loader, None, criterion)

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
        f"val loss {va_loss:.4f} acc {va_acc:.4f}"
    )

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

# STEP 5. Load the test images
test_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))
x_test = load_test_images(test_paths, (IMG_W, IMG_H)).to(DEVICE)

# STEP 6. Switch CNN to evaluation mode
model.eval()

# STEP 7. Run CNN with disabled gradients (for better speed and memory efficiency)
with torch.no_grad():
    # 7.1. Infer the logits for the model (i.e., the unnormalized output of the final linear layer)
    logits = model(x_test)
    # 7.2. Convert logits to probabilities via sigmoid function
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    # 7.3. Infer predicted classes with probabilities at 0.5 or larger
    preds = (probs >= 0.5).astype(int)

################################################################################
# PRINT OUTPUTS

print("Predicted probabilities:", probs)
# EXAMPLE OUTPUT:  Predicted probabilities: [0.91, 0.12]

print("Predicted classes (0 or 1):", preds)
# EXAMPLE OUTPUT:  Predicted classes (0 or 1): [1, 0]

# Print training labels to mapping (use folder name as index names)
print("Training class_to_idx:", full_ds.class_to_idx)
# EXAMPLE OUTPUT:  Training class_to_idx: {'dicot': 0, 'monocot': 1}

################################################################################
# EXTENDED USAGE

# If labels for test images are available, model accuracy can be calculated:
# y_test = np.array([...], dtype=np.int64)
# test_acc = (preds == y_test).mean()
# print("Test accuracy:", test_acc)
