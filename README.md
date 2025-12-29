# Image classification for plant anatomy using supervised machine learning
Image classification for classifying root cross sections as dicot and monocot using convolutional neural networks

## Summary
This script trains and executes a small convolutional neural network (CNN) to classify images of root cross-section as **dicot** or **monocot**. The network consists of two convolutional blocks for feature extraction followed by fully connected layers that output a single logit for binary classification.

Training images are loaded from class-named subfolders, split into training and validation sets, and then algorithimcally optimized and scored. Test images are loaded separately from a flat directory and used for inference only.

## Expected folder structure

### Training data (`TRAIN_DIR`)
```
data/train/
├── dicot/
│ └── *.jpg
└── monocot/
  └── *.jpg
```
### Test data (`TEST_DIR`)
```
data/test/
  └── *.jpg
```

## How to run it
1. Install dependencies:
```bash
pip install torch torchvision numpy opencv-python
```
2. Verify that variables `TRAIN_DIR` and `TEST_DIR` in the code point to the correct locations.
3. Run the script:
```bash
python3 RootXS_identification_via_CNN_image_classification.py
```
