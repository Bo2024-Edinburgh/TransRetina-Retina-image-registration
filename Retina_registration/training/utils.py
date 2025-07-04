import os
import random
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict


def setup_logging(log_dir: str):
    """
    Set up logging to both a file and the console.

    Args:
        log_dir (str): Directory where the log file will be stored.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),  # Log to file
            logging.StreamHandler()                         # Log to stdout
        ]
    )

def seed_everything(seed: int = 42):
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The seed value to be set. Default is 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)     # Ensure reproducible Python hashing
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For complete determinism (can reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    logging.info(f"🌱 Random seed set to: {seed}")

def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file for training.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict: Parsed configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Enable mixed precision by default if not explicitly defined
    if 'mixed_precision' not in config['train']:
        config['train']['mixed_precision'] = True

    logging.info(f"📋 Configuration loaded from: {config_path}")
    return config
