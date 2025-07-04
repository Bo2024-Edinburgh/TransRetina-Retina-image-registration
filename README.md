# Deep Registration Model

![Model](../images/model.png)

## Overview

This project implements a deep learning based image registration framework, integrating deformation field estimation and keypoint-based guidance. The architecture supports smoothness constraints and multi-component loss functions (MSE, NCC, keypoint alignment), and can be adapted to various medical or remote sensing datasets.

---

## Training

### 1. Generate image pairs

Prepare your dataset as aligned or misaligned image pairs. This can involve manual pairing or using domain-specific augmentation to generate deformation.

---

### 2. Extract keypoints

Use a keypoint detection module (not included here) to extract keypoints for all training images.  
Save the keypoints in the expected annotation format (e.g., `.txt` or `.json` files).

---

### 3. Configure `train.yaml`

Edit the `train.yaml` configuration file to:

- set the paths to your images and annotations (`train_image_dir` and `anno_file_dir`),
- adjust the model input size and training parameters.

---

### 4. Start training

Run:

```bash
python train.py --config train.yaml
