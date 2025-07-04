import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def apply_full_transform(img):
    """
    Apply a sequence of data augmentation transforms to the input image,
    including affine, elastic deformation, and additive Gaussian noise.
    """
    img = affine_transform(img)
    img = elastic_transform(img)
    img = add_noise(img)
    return img

def affine_transform(img):
    """
    Perform random affine transformation on the image,
    including rotation, scaling, and translation.
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-10, 10)  # random rotation angle
    scale = random.uniform(0.9, 1.1)  # random scaling
    tx = random.uniform(-0.05 * w, 0.05 * w)  
    ty = random.uniform(-0.05 * h, 0.05 * h) 

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

def elastic_transform(img, alpha=50, sigma=6):
    """
    Apply elastic deformation to the image based on a random displacement field
    smoothed by a Gaussian filter.

    Parameters:
    - alpha: scaling factor for displacement
    - sigma: Gaussian kernel standard deviation
    """
    h, w = img.shape[:2]
    random_state = np.random.RandomState(None)

    dx = cv2.GaussianBlur((random_state.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

def add_noise(img, std=5):
    """
    Add Gaussian noise with specified standard deviation to the image.
    """
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# === Main Function ===
def generate_pairs_with_uniform_naming(input_dir, output_dir, num_pairs_per_image=2):
    """
    Generate fixed-moving image pairs with consistent zero-padded filenames.

    Parameters:
    - input_dir: path to the directory containing original images
    - output_dir: path to save generated image pairs under 'fixed' and 'moving'
    - num_pairs_per_image: number of pairs to generate per input image
    """
    fixed_dir = os.path.join(output_dir, "fixed")
    moving_dir = os.path.join(output_dir, "moving")
    os.makedirs(fixed_dir, exist_ok=True)
    os.makedirs(moving_dir, exist_ok=True)

    image_list = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])

    counter = 0  # global uniform counter for naming

    for fname in tqdm(image_list, desc="Processing images"):
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[Skipped] Cannot read image: {fname}")
            continue

        for _ in range(num_pairs_per_image):
            fixed_img = apply_full_transform(img)
            moving_img = apply_full_transform(img)

            filename = f"{counter:06d}.jpg"
            cv2.imwrite(os.path.join(fixed_dir, filename), fixed_img)
            cv2.imwrite(os.path.join(moving_dir, filename), moving_img)
            counter += 1

    print(f"\n Generated {counter} pairs with filenames from 000000.jpg to {counter-1:06d}.jpg")


input_dir = "dataset/training_set/images"
output_dir = "dataset/training_set/pair_images"
generate_pairs_with_uniform_naming(input_dir, output_dir, num_pairs_per_image=2)
