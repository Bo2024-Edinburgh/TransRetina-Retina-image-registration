import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from PIL import Image
import random
import yaml
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import InterpolationMode
import cv2




"""
 生成一个与图像尺寸相同的二值掩码图，其中关键点的位置被标记为 1.0。
 points: 关键点的坐标数组，形状为 (num_points, 2)。
 img_size: 图像的尺寸，形状为 (height, width)。
 返回一个与图像尺寸相同的二值掩码图，其中关键点的位置被标记为 1.0。
"""
def generate_guide_map(points, img_size):
    mask_guide = np.zeros((img_size, img_size))
    
    # 遍历关键点数组，将每个关键点的位置在掩码图上标记为 1.0。
    for point in points:  
        x,y = point
        if(int(x) > img_size or int(y)> img_size):
            continue
        mask_guide[int(y), int(x)] = 1.0
    return mask_guide


def split_dataset(image_dir, split_ratio=0.85, seed=42):
    random.seed(seed)
    files = os.listdir(image_dir)
    ids = sorted(set(f.split('_')[0] for f in files if f.endswith('_fixed.jpg')))
    random.shuffle(ids)
    split_index = int(len(ids) * split_ratio)
    return ids[:split_index], ids[split_index:]



class My_Dataset_Points(data.Dataset):
    def __init__(self, list_IDs, config):
        self.list_IDs = list_IDs
        self.config = config
        self.img_dir = config['train']['train_image_dir']
        self.anno_dir = config['train']['anno_file_dir']
        self.img_size = config['train']['model_image_width']
        self.use_score_map = config['train'].get('PAG_map', True)
        self.point_select = config['train'].get('point_select', 2)

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        # 高斯核
        k_size = 8
        sigma = 3
        kernel = cv2.getGaussianKernel(k_size, sigma)
        self.gaussian_kernel = kernel @ kernel.T

        if self.use_score_map:
            dis_map = cv2.imread(config['train']['dis_map_path'], 0)  # [96, 96]
            dis_map = cv2.resize(dis_map, (96, 96), interpolation=cv2.INTER_LINEAR)
            dis_map = torch.tensor(dis_map / 255, dtype=torch.float32)  # [96, 96]
            dis_map = dis_map.view(1, -1).repeat(9216, 1)  # [9216, 9216]
            self.gt_dis_map = dis_map
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        fx_path = os.path.join(self.img_dir, f"{ID}_fixed.jpg")
        mv_path = os.path.join(self.img_dir, f"{ID}_moving.jpg")
        anno_path = os.path.join(self.anno_dir, f"{ID}.txt")

        fixed_pts, moving_pts = [], []
        with open(anno_path, 'r') as f:
            for line in f:
                x_fix, y_fix, x_mov, y_mov = map(float, line.strip().split())
                fixed_pts.append([x_fix, y_fix])
                moving_pts.append([x_mov, y_mov])

        fixed_pts = np.array(fixed_pts)
        moving_pts = np.array(moving_pts)

        fixed_img = Image.open(fx_path).convert("L")
        moving_img = Image.open(mv_path).convert("L")
        fixed_np = np.array(fixed_img)

        fixed_tensor = self.transforms(fixed_img)   # [1, H, W]
        moving_tensor = self.transforms(moving_img)

        # 梯度排序
        if self.point_select == 2:
            gx = cv2.Sobel(fixed_np, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(fixed_np, cv2.CV_64F, 0, 1)
            mag = np.sqrt(gx**2 + gy**2)
            grads = [mag[int(y), int(x)] for x, y in fixed_pts]
            sorted_idx = np.argsort(grads)[::-1]
            fixed_pts = fixed_pts[sorted_idx]
            moving_pts = moving_pts[sorted_idx]

        fixed_pts = fixed_pts[:60]
        moving_pts = moving_pts[:60]

        mask = generate_guide_map(fixed_pts, self.img_size)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.use_score_map:
            fix_map = generate_score_map(fixed_pts, self.gaussian_kernel, self.img_size)
            mov_map = generate_score_map(moving_pts, self.gaussian_kernel, self.img_size)

            resize = transforms.Resize((96, 96), interpolation=InterpolationMode.BILINEAR)
            fix_map = resize(fix_map).squeeze()  # [96,96]
            mov_map = resize(mov_map).squeeze()  # [96,96]

            fix_map = fix_map.view(1, -1).t()    # (9216, 1)
            mov_map = mov_map.view(1, -1)        # (1, 9216)

            score_map = torch.matmul(fix_map, mov_map)  # (9216, 9216)

            score_map = 0.5 * (fix_map * mov_map + self.gt_dis_map)
        else:
            score_map = torch.zeros((96, 96), dtype=torch.float32)

        # ✅ 新增：生成 Canny 边缘图作为 attention 引导
        edge_map_np = cv2.Canny(fixed_np, threshold1=50, threshold2=150)  # [H, W]
        edge_map = torch.from_numpy(edge_map_np).float().unsqueeze(0) / 255.0  # [1, H, W]

        return fixed_tensor, moving_tensor, \
               torch.tensor(fixed_pts, dtype=torch.float32), \
               torch.tensor(moving_pts, dtype=torch.float32), \
               mask, score_map, edge_map


def generate_score_map(fixed_point, kennel_size, size=768):
    # 生成score map
     ## 生成attention score map
    fix_map = generate_guide_map(fixed_point , size)
    # 使用高斯卷积来模糊图像
    fix_map = cv2.filter2D(fix_map, -1, kennel_size)
    fix_map = torch.tensor(fix_map).to(torch.float32).unsqueeze(0)
    max_value = torch.max(fix_map)
    min_value = torch.min(fix_map)
    # 进行归一化
    fix_map = (fix_map - min_value) / (max_value - min_value)
    return fix_map