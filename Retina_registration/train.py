import os
import datetime
import logging
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from training.data_utils import create_data_loaders
from training.trainer import TransRetinaTrainer
from training.utils import setup_logging, seed_everything, load_config


def main():
    """
    Main training function that prepares configuration, sets up logging,
    initializes the model and trainer, and starts the training loop.
    """
    # Path to configuration file
    config_path = 'config/train_config/train_low.yaml'
    
    # Set random seeds for reproducibility
    seed_everything(42)
    
    # Load training configuration
    config = load_config(config_path)
    
    # Create base saving directory
    save_dir = Path(config['train']["save_files_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model-specific directory
    model_save_dir = save_dir / config['train']["model_save_prefix"]
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Create logging directory
    log_dir = model_save_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging to file and console
    setup_logging(str(log_dir))
    
    # Log training start info
    current_time = datetime.datetime.now()
    logging.info("=" * 50)
    logging.info("🚀 VoxelMorph Training Started")
    logging.info(f"⏰ Start time: {current_time}")
    logging.info("=" * 50)
    
    # Determine device (CPU or GPU)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    logging.info(f"🖥️  Device: {device}")
    
    # Initialize trainer, including model and optimizer setup
    input_dims = (1, config['train']["model_image_width"], config['train']["model_image_width"])
    trainer = TransRetinaTrainer(config, input_dims, use_gpu)
    
    # Prepare training and validation data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize TensorBoard logger
    log_writer = SummaryWriter(
        log_dir=save_dir / "logs" / config['train']["model_save_prefix"]
    )
    
    try:
        # Begin training
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['train']["train_epochs"],
            log_writer=log_writer
        )
        
    except KeyboardInterrupt:
        logging.info("⚠️  Training interrupted by user")
        
    except Exception as e:
        logging.error(f"❌ Training failed with error: {str(e)}")
        raise
        
    finally:
        # Clean up TensorBoard writer
        log_writer.close()
        logging.info("📊 Tensorboard writer closed")


if __name__ == "__main__":
    main()
