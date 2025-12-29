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

########################################
# GLOBAL VARIABLES
IMG_W, IMG_H = 150, 150  # target size of images
BATCH_SIZE = 10  # batch_size controls how many images the dataloader loads per batch
EPOCHS = 10
TRAIN_DIR = "~/git/michaelgruenstaeudl_CNN_monocot_dicot_rootXS/data/train/"
TEST_DIR = "~/git/michaelgruenstaeudl_CNN_monocot_dicot_rootXS/data/test/"
VAL_SPLIT = 0.2  # Fraction of training data used for validation
SEED = 42  # Douglas Adams would be proud :-)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

################################################################################
# TRAINING THE MODEL
################################################################################

########################################
# GENERATE TRAINING AND VALIDATION DATA
# STEP 1. Automatic transformation of images for training data:
#         resizing images to a fixed target size and
#         rescaling pixel values to range
train_tfms = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),  # normalizes pixel values to range [0, 1] (by multiplying by 1/255)
])

# STEP 2. Read images from subfolders and assign class labels based on folder names
# ImageFolder expects: data/train/<class_name>/*.jpg, with classes [monocot, dicot]
full_ds = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tfms)

# STEP 3. Splitting data into training and validation dataset
n = len(full_ds)
indices = np.random.permutation(n)  # Create a permutation of indices [0..n-1]
val_size = int(round(n * VAL_SPLIT))  # Infer VAL_SPLIT fraction size

# Define indices for training dataset and then pick training data based on indices
train_idx = indices[val_size:]
train_ds = Subset(full_ds, train_idx.tolist())

# Define indices for validation dataset and then pick validation data based on indices
val_idx = indices[:val_size]
val_ds = Subset(full_ds, val_idx.tolist())

train_loader = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=True, # shuffle: randomizes training batches each epoch
                          num_workers=0,
                          pin_memory=(DEVICE.type == "cuda") # Using NVIDIA GPU to accelerate computation, if GPU available
                          )
val_loader = DataLoader(val_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=(DEVICE.type == "cuda")
                        )

########################################
# FUNCTIONS FOR TRAINING AND EVALUATING THE DATA

def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
''' Helper function to convert logits to probabilities using a sigmoid function,
    applies a threshold of 0.5 to infer predicted classes, and returns
    the mean accuracy across the given batch.
'''
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds.eq(y)).float().mean().item()

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    ) -> Tuple[float, float]:
''' Ths function specifies both CNN training and CNN validation:
        - If an optimizer is provided, it runs in training mode, does backward(), and updates weights.
        - If no optimizer, it runs in evaluation mode (no gradient updates).
    Input:
        1. nn.Module: this is the neural network being trained or evaluated;
            during training: parameters are updated; during validation: only used for forward passes
        2. DataLoader: supplies the image and label data batch by batch
        3. torch.optim.Optimizer or None: determines whether the function performs training or evaluation;
            during training (optimizer provided): gradients are computed, model weights are updated
            during evaluation (no optimizer): no backpropagation, no weight updates
        4. LossFunction: nn.BCEWithLogitsLoss
    Output:
        1. avg_loss: indicates how well prediction matches target over all batches in the epoch; lower is better
        2. avg_accuracy: specifies fraction of correct predictions over all batches in the epoch; higher is better
'''

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    # Supplying the image and label data batch by batch
    for x, y in loader:
        # x is a batch of images with shape (batch_size, 3, H, W)
        x = x.to(DEVICE, non_blocking=True)

        # y is a batch of binary labels with shape (batch_size,)
        y = y.to(DEVICE, non_blocking=True).float().view(-1, 1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += binary_accuracy_from_logits(logits.detach(), y)
        batches += 1

    return total_loss / max(batches, 1), total_acc / max(batches, 1)


########################################
# ACTUALLY TRAIN THE DATA

# STEP 4. Instantiate the model and move it to DEVICE (GPU if available)
model = RootCNN().to(DEVICE)

# STEP 5. Specify the loss function to measure how well the predicted probabilities match the binary targets
criterion = nn.BCEWithLogitsLoss()  # better than sigmoid activation and binary cross-entropy loss function in Keras

# STEP 6. Specify the output optimization using RMSprop
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)  # roughly analogous to Keras "rmsprop"

# STEP 7 (ML OCCRUS HERE). Iterating over all samples (i.e. epoch) to calculate loss function and perform optimization
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
# EVALUATING TEST DATA USING TRAINED MODEL
################################################################################

########################################
# LOCATE THE TEST IMAGE DATA ON DISK, RESIZE IMAGES, AND RUN CODE

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
        img = img.astype(np.float32) / 255.0  # rescaling pixel values to range [0, 1] (by multiplying by 1/255)
        img = np.transpose(img, (2, 0, 1))    # HWC -> CHW (PyTorch expects channels (C) first)
        imgs.append(img)
    if not imgs:
        raise RuntimeError(f"No test images present in: {TEST_DIR}")
    # Returning images as a single tensor of shape (N, 3, H, W)
    return torch.from_numpy(np.stack(imgs, axis=0))

# STEP 8. Load the test images
test_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))
x_test = load_test_images(test_paths, (IMG_W, IMG_H)).to(DEVICE)

# STEP 9. Switch CNN to evaluation mode
model.eval()

# STEP 10. Run CNN with disabled gradients (for better speed and memory efficiency)
with torch.no_grad():
    # 10.1. Infer the logits for the model (i.e., the unnormalized output of the final linear layer)
    logits = model(x_test)
    # 10.2. Convert logits to probabilities via sigmoid function
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    # 10.3. Infer predicted classes with probabilities at 0.5 or larger
    preds = (probs >= 0.5).astype(int)

########################################
# PRINT OUTPUTS

print("Predicted probabilities:", probs)
# EXAMPLE OUTPUT:  Predicted probabilities: [0.91, 0.12]

print("Predicted classes (0 or 1):", preds)
# EXAMPLE OUTPUT:  Predicted classes (0 or 1): [1, 0]

# Print training labels to mapping (use folder name as index names)
print("Training class_to_idx:", full_ds.class_to_idx)
# EXAMPLE OUTPUT:  Training class_to_idx: {'dicot': 0, 'monocot': 1}

########################################
# EXTENDED USAGE

# If labels for test images are available, model accuracy can be calculated:
# y_test = np.array([...], dtype=np.int64)
# test_acc = (preds == y_test).mean()
# print("Test accuracy:", test_acc)
