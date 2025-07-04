import os
import logging
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from dataset.dataset_process import My_Dataset_Points, split_dataset, custom_collate_fn


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders based on the given configuration.

    Args:
        config (Dict): Configuration dictionary containing training and data parameters.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation DataLoaders.
    """
    train_config = config['train']

    # Determine optimal number of workers (max 8 or number of available CPU cores)
    optimal_workers = min(8, os.cpu_count())

    # Split dataset into training and validation sets
    train_ids, val_ids = split_dataset(train_config["train_image_dir"])
    partition = {'train': train_ids, 'validation': val_ids}

    logging.info(f"📂 Dataset split - Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Initialize training and validation datasets
    train_dataset = My_Dataset_Points(partition['train'], config)
    val_dataset = My_Dataset_Points(partition['validation'], config)

    # Set parameters for the training DataLoader
    train_params = {
        'batch_size': train_config["batch_size"],
        'shuffle': True,
        'num_workers': optimal_workers,
        'pin_memory': True,               # Accelerate data transfer to GPU
        'prefetch_factor': 2,             # Prefetch batches in advance
        'persistent_workers': True,       # Reuse worker processes between epochs
        'drop_last': True,                # Drop the last incomplete batch
        'collate_fn': custom_collate_fn,  # Custom collate function for complex samples
        'worker_init_fn': lambda x: np.random.seed(42 + x)  # Ensure reproducible worker behavior
    }

    # Set parameters for the validation DataLoader
    val_params = {
        'batch_size': min(4, train_config["batch_size"]),  # Use a smaller batch size for validation
        'shuffle': False,
        'num_workers': max(1, optimal_workers // 2),
        'pin_memory': True,
        'prefetch_factor': 2,
        'collate_fn': custom_collate_fn,  # Same custom collate function for consistency
        'persistent_workers': True
    }

    # Construct DataLoaders
    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **val_params)

    return train_loader, val_loader
