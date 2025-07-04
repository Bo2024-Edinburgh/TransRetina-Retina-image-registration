import numpy as np
import torch


class keypoints_process:
    def __init__(self) -> None:
        pass

    """
    Reads original keypoints from a text file and splits them into
    keypoints for moving (mov) and fixed (fix) images.

    Args:
        gt_txt_file (str): path to the keypoint coordinate text file.

    Returns:
        mov (np.ndarray): keypoints on the moving image, shape (N, 2)
        fix (np.ndarray): keypoints on the fixed image, shape (N, 2)
    """
    def keypoint_read_original(self, gt_txt_file):
        points_gd = np.loadtxt(gt_txt_file)
        fix = np.zeros([len(points_gd), 2])
        mov = np.zeros([len(points_gd), 2])
        fix[:, 0] = points_gd[:, 0]  # dst -> refer (fixed)
        fix[:, 1] = points_gd[:, 1]
        mov[:, 0] = points_gd[:, 2]  # raw -> query (moving)
        mov[:, 1] = points_gd[:, 3]
        return mov, fix

    """
    Scale keypoint coordinates batch-wise.

    Typically used when:
    - 'old': scale from original image size to network input size (e.g. 2912 -> 768)
    - 'new': reverse scaling for prediction recovery.

    Args:
        config (dict): configuration dictionary containing sizes.
        points1 (np.ndarray): points to scale, shape (N, 2)
        type (str): 'old' for original -> input, 'new' for input -> original.

    Returns:
        np.ndarray: scaled points.
    """
    def keypoint_scale(self, config, points1, type= "old"):
        if (type == "old"):
            scale_w = config["PREDICT"]["image_original_size"] / config["PREDICT"]["model_image_width"] 
            scale_h = config["PREDICT"]["image_original_size"] / config["PREDICT"]["model_image_width"]  
        else:
            scale_w = config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"]
            scale_h = config["PREDICT"]["model_image_width"] / config["PREDICT"]["image_original_size"]
        points1[:, 0] = (points1[:, 0] * scale_w)
        points1[:, 1] = (points1[:, 1] * scale_h)
        return points1

    """
    Optimized sampling function for training, with a reduced search area
    (25 instead of 40) to accelerate when dealing with small deformation fields.

    Args:
        raw_point (np.ndarray): keypoints on moving image, shape (N, 2)
        flow (torch.Tensor): deformation field, shape [1, 2, H, W]

    Returns:
        np.ndarray: predicted destination points, shape (N, 2)
    """
    def points_sample_nearest_train(self, raw_point, flow):
        dst_pred = np.zeros([raw_point.shape[0], 2])
        H, W = flow.shape[2], flow.shape[3]  # height and width
        for index_p in range(raw_point.shape[0]):
            row_index = (raw_point[index_p, 0])  # x-coordinate
            col_index = (raw_point[index_p, 1])  # y-coordinate
            search_area = 25
            min_distance = 10000.0
            for x_search in range(row_index - search_area, row_index + search_area):
                for y_search in range(col_index - search_area, col_index + search_area):
                    if x_search < 0 or x_search >= W or y_search < 0 or y_search >= H:
                        continue  # skip out-of-bound indices
                    distance = np.abs(row_index - int(flow[0,1,y_search,x_search].item())) \
                             + np.abs(col_index - int(flow[0,0,y_search,x_search].item()))
                    if (distance < min_distance):
                        dst_pred[index_p, 0] = x_search
                        dst_pred[index_p, 1] = y_search
                        min_distance = distance
        return dst_pred
