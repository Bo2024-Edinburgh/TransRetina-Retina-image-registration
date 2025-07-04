import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
import time
import gc
import json
import datetime
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

from training.metrics import TrainingMetrics, ValidationMetrics
from training.model_utils import print_model_architecture, print_layer_statistics, ModelSaver, LearningRateScheduler
from utils.keypoints import keypoints_process  
from utils import loss as L


class TransRetinaTrainer:
    """
    Main VoxelMorph training class with mixed precision support and gradient accumulation.
    
    This trainer implements a comprehensive training pipeline for retinal image registration
    using transformer-based neural networks with advanced optimization techniques including:
    - Mixed precision training for memory efficiency
    - Gradient accumulation for effective large batch training
    - Dynamic learning rate scheduling
    - Multi-component loss functions (similarity, smoothness, keypoint alignment)
    - Comprehensive validation with both loss and distance metrics
    """
    
    def __init__(self, config: Dict, input_dims: Tuple[int, ...], use_gpu: bool = True):
        """
        Initialize the TransRetinaTrainer with configuration and setup.
        
        Args:
            config: Training configuration dictionary containing model and training parameters
            input_dims: Input tensor dimensions (channels, height, width)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.config = config
        self.train_config = config['train']
        self.input_dims = input_dims
        self.device = self._setup_device(use_gpu)
        
        # Mixed precision setup for memory efficiency and speed
        self.use_mixed_precision = self.train_config.get("mixed_precision", True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)
        
        # Initialize model and components
        self.model = self._build_model()
        
        # Print model architecture before training
        self._print_model_info()

        # Gradient accumulation configuration for effective large batch training
        self.accumulation_steps = self.train_config.get('gradient_accumulation_steps', 1)
        self.current_step = 0
        self.warmup_steps = self.train_config.get('warmup_steps', 0)

        # # Dynamic weight adjustment for loss components
        # self.initial_weights = self.train_config["loss_weights"].copy()
        # self.adaptive_weights = True
        # self.smooth_loss_threshold = 0.001  # Smooth loss threshold
        

        self.optimizer = self._build_optimizer()
        self.loss_functions = self._build_loss_functions()
        self.lr_scheduler = self._build_dynamic_scheduler()
        # self.lr_scheduler = LearningRateScheduler(
        #     self.optimizer, 
        #     self.train_config["lr"],
        #     self.train_config["ir_epochs"]
        # )

        # Loss history tracking for monitoring training progress
        self.train_loss_history = []
        self.val_loss_history = []
        self.loss_save_path = Path(self.train_config["save_files_dir"]) / self.train_config["model_save_prefix"] / "loss_history.json"

        # Pre-create zero tensor to avoid repeated tensor creation
        self.zero_tensor = torch.tensor(0.0, device=self.device)
        
        logging.info(f"📈 Gradient accumulation steps: {self.accumulation_steps}")
        if self.warmup_steps > 0:
            logging.info(f"🔥 Warmup steps: {self.warmup_steps}")

        # Training utilities initialization
        self.keypoints_processor = keypoints_process()
        model_save_dir = Path(self.train_config["save_files_dir"]) / self.train_config["model_save_prefix"]
        self.model_saver = ModelSaver(
            str(model_save_dir),
            self.train_config["model_save_prefix"],
            self.train_config["save_epochs"]
        )
        
        # Training state variables
        self.current_epoch = 0
        self.image_size = self.train_config["model_image_width"]
        
        logging.info(f"🚀 TransRetinaTrainer initialized on device: {self.device}")
        logging.info(f"⚡ Mixed precision training: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """
        Setup computation device with proper GPU memory management.
        
        Args:
            use_gpu: Whether to attempt GPU usage
            
        Returns:
            torch.device: Configured device for computation
        """
        if use_gpu and torch.cuda.is_available():
            device = torch.device(self.train_config.get("device", "cuda"))
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
        return device
    
    def _build_model(self) -> nn.Module:
        """
        Build and initialize the RetinalTransUNet model.
        
        Returns:
            nn.Module: Configured model moved to appropriate device
        """
        model = RetinalTransUNet(
            self.input_dims[0] * 2, 
            self.config, 
            self.device != torch.device("cpu"), 
            self.device
        )
        return model.to(self.device)
    
    def _print_model_info(self):
        """Print comprehensive model architecture and statistics information."""
        print_model_architecture(self.model, self.input_dims, self.device)
        print_layer_statistics(self.model)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """
        Build Adam optimizer with AMSGrad for stable training.
        
        Returns:
            optim.Optimizer: Configured optimizer
        """
        return optim.Adam(
            self.model.parameters(), 
            lr=self.train_config["lr"], 
            amsgrad=True
        )
    
    def _build_loss_functions(self) -> Dict:
        """
        Build dictionary of loss functions for different training objectives.
        
        Returns:
            Dict: Dictionary containing all loss function instances
        """
        loss_config = self.train_config
        
        return {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss(),
            'key_mse': nn.MSELoss(reduction="sum"),
            'smooth': L.smooth_loss(),
            'ncc': L.ncc_loss_fun(self.device, win=loss_config["window_size"]),
        }

    def _build_dynamic_scheduler(self):
        """
        Build dynamic learning rate scheduler based on configuration.
        
        Returns:
            Learning rate scheduler instance (CosineAnnealingLR or custom LearningRateScheduler)
        """
        scheduler_type = self.train_config.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=50,           # 50 epochs per cycle
                eta_min=1e-6        # Minimum learning rate
            )
        else:
            # Keep original learning rate scheduler as default
            return LearningRateScheduler(
                self.optimizer, 
                self.train_config["lr"],
                self.train_config["ir_epochs"]
            )

    def _apply_warmup(self, step: int):
        """
        Apply learning rate warmup for stable training initialization.
        
        Args:
            step: Current training step
        """
        if step < self.warmup_steps:
            lr_scale = min(1.0, float(step + 1) / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.train_config["lr"] * lr_scale

    def _save_loss_history(self):
        """Save loss history to JSON file for training monitoring and analysis."""
        try:
            loss_history = {
                'train_history': self.train_loss_history,
                'val_history': self.val_loss_history,
                'config': {
                    'total_epochs': len(self.train_loss_history),
                    'last_epoch': self.current_epoch,
                    'save_time': datetime.datetime.now().isoformat()
                }
            }
            
            # Ensure directory exists
            self.loss_save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to JSON file
            with open(self.loss_save_path, 'w', encoding='utf-8') as f:
                json.dump(loss_history, f, indent=2, ensure_ascii=False)
            
            logging.info(f"💾 Loss history saved to: {self.loss_save_path}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save loss history: {str(e)}")

    def _compute_losses(self, registered_image, batch_fixed, deformation_matrix, 
                       mask_guide) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity and smoothness losses for image registration.
        
        Args:
            registered_image: Transformed moving image
            batch_fixed: Target fixed image
            deformation_matrix: Computed deformation field
            mask_guide: Mask for guided registration
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Similarity loss and smoothness loss
        """
        loss_type = self.train_config["loss"]
        
        # Compute similarity loss based on configuration
        if loss_type == "l2":
            sim_loss = self.loss_functions['mse'](registered_image, batch_fixed)
        elif loss_type == "ncc":
            sim_loss = self.loss_functions['ncc'].ncc_loss(registered_image, batch_fixed, mask_guide)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Compute smoothness loss with numerical stability checks
        if self.train_config["loss_with_smooth"]:
            smooth_loss = self.loss_functions['smooth'].smooothing_loss_2d(deformation_matrix)
            
            # Numerical stability check
            if torch.isnan(smooth_loss) or torch.isinf(smooth_loss):
                smooth_loss = self.zero_tensor
                logging.warning("Smooth loss is NaN/Inf, setting to zero")
            else:
                # Fixed loss clipping, independent of epoch
                smooth_loss = torch.clamp(smooth_loss, max=0.002)  # Fixed upper limit
        else:
            smooth_loss = self.zero_tensor
        
        return sim_loss, smooth_loss

    def _compute_keypoint_loss(self, deformation_matrix, fix_points, mov_points) -> torch.Tensor:
        """
        Compute keypoint alignment loss for spatial correspondence.
        
        Args:
            deformation_matrix: Computed deformation field
            fix_points: Fixed image keypoints
            mov_points: Moving image keypoints
            
        Returns:
            torch.Tensor: Keypoint alignment loss
        """
        grid_integrated = self.model.transformer.grid + deformation_matrix
        batch_size = fix_points.shape[0]

        if batch_size == 0 or fix_points.numel() == 0:
            return self.zero_tensor

        # Vectorized processing for all batches
        grid_flat = grid_integrated.view(batch_size, -1, 2)  # B x (H*W) x 2

        # Batch index computation - ensure within valid range
        fix_indices = (fix_points[..., 1] * self.image_size + fix_points[..., 0]).long()
        fix_indices = torch.clamp(fix_indices, 0, self.image_size * self.image_size - 1)
        
        # Batch sampling - use efficient gather operation
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, fix_indices.size(1))
        mov_pred = grid_flat[batch_indices, fix_indices]  # B x N x 2
        
        # Coordinate conversion: y,x -> x,y
        mov_pred = mov_pred[..., [1, 0]]
        
        # Batch normalization and loss computation
        mov_pts_norm = mov_points.float() / self.image_size
        mov_pred_norm = mov_pred / self.image_size
        
        # # Primarily use original smooth_l1_loss with slight distance awareness
        # base_loss = F.smooth_l1_loss(mov_pts_norm, mov_pred_norm, reduction='mean', beta=0.1)
        
        # # Slight distance weighting (optional)
        # if hasattr(self, 'current_epoch') and self.current_epoch > 20:
        #     distances = torch.norm(mov_pts_norm - mov_pred_norm, dim=-1)
        #     distance_weight = 1.0 + 0.1 * distances.detach()  # Very small weight adjustment
        #     weighted_loss = (distance_weight * distances).mean()
        #     keypoint_loss = 0.8 * base_loss + 0.2 * weighted_loss
        # else:
        #     keypoint_loss = base_loss

        # Use more stable loss function
        keypoint_loss = F.smooth_l1_loss(mov_pts_norm, mov_pred_norm, reduction='mean', beta=0.1)
        
        return keypoint_loss

    def _compute_distance_metrics(self, deformation_matrix, fix_points, mov_points) -> Tuple[float, float]:
        """
        Compute distance-based validation metrics for registration quality assessment.
        
        Args:
            deformation_matrix: Computed deformation field
            fix_points: Fixed image keypoints
            mov_points: Moving image keypoints
            
        Returns:
            Tuple[float, float]: Average error and maximum error in pixels
        """
        # If too many points, randomly sample a subset to reduce computation
        if fix_points.shape[1] > 50:  # If keypoints exceed 50
            num_samples = 50
            indices = torch.randperm(fix_points.shape[1])[:num_samples]
            fix_points = fix_points[:, indices]
            mov_points = mov_points[:, indices]
        
        # Keep original computation logic but with reduced data volume
        flow_pred = self.model.transformer.grid + deformation_matrix
        
        mov_points_np = mov_points.detach().cpu().numpy()
        fix_points_np = fix_points.detach().cpu().numpy()

        # Handle batch dimension
        if mov_points_np.ndim == 3:  # (batch, num_points, 2)
            mov_points_np = mov_points_np[0]  # Take first sample
        if fix_points_np.ndim == 3:
            fix_points_np = fix_points_np[0]
        
        # Ensure integer type coordinates
        mov_points_np = mov_points_np.astype(np.int32)
        fix_points_np = fix_points_np.astype(np.int32)
        
        # Key modification: keep flow_pred as torch tensor format, keypoints.py needs this format
        dst_pred = self.keypoints_processor.points_sample_nearest_train(
            raw_point=mov_points_np, flow=flow_pred
        )
        
        # Calculate distances
        distances = np.sqrt(np.sum((fix_points_np - dst_pred) ** 2, axis=1))
        avg_error = distances.mean()
        max_error = distances.max()
        
        return avg_error, max_error
    
    def train_step(self, batch_data) -> Tuple[float, ...]:
        """
        Execute one training step with mixed precision training.
        
        Args:
            batch_data: Batch of training data including images, keypoints, and masks
            
        Returns:
            Tuple[float, ...]: Training metrics (total_loss, dice_score, sim_loss, smooth_loss, keypoint_loss, attention_loss)
        """
        # Unpack batch data
        batch_fixed, batch_moving, fix_points, mov_points, mask_guide, gt_score_map, edge_map = batch_data
        
        # Move data to device with non-blocking transfer for efficiency
        batch_fixed = batch_fixed.to(self.device, non_blocking=True)
        batch_moving = batch_moving.to(self.device, non_blocking=True)
        fix_points = torch.round(fix_points).to(torch.int32).to(self.device, non_blocking=True)
        mov_points = mov_points.to(torch.float32).to(self.device, non_blocking=True)
        mask_guide = mask_guide.to(self.device, non_blocking=True)
        gt_score_map = gt_score_map.to(self.device, non_blocking=True)
        edge_map = edge_map.to(self.device, non_blocking=True) if edge_map is not None else None
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            # Forward pass through model based on configuration
            if self.train_config["Em_map"]:
                if self.train_config["PAG_map"]:
                    registered_image, deformation_matrix, attention_score = self.model(
                        batch_moving, batch_fixed, edge_map
                    )
                    attention_loss = F.smooth_l1_loss(gt_score_map, attention_score, reduction="mean", beta=1.0)
                else:
                    registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, edge_map)
                    attention_loss = torch.tensor(0.0, device=self.device)
            else:
                registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, mask_guide)
                attention_loss = torch.tensor(0.0, device=self.device)
            
            # Compute individual loss components
            sim_loss, smooth_loss = self._compute_losses(
                registered_image, batch_fixed, deformation_matrix, mask_guide
            )
            
            keypoint_loss = self._compute_keypoint_loss(deformation_matrix, fix_points, mov_points)
            
            # Compute weighted total loss
            weights = self.train_config["loss_weights"]
            total_loss = (weights[0] * sim_loss + 
                         weights[1] * smooth_loss + 
                         weights[2] * keypoint_loss)
            
            if self.train_config["PAG_map"]:
                total_loss += weights[3] * attention_loss
        
        # Backward pass with gradient scaling for mixed precision
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Compute additional metrics without gradients
        with torch.no_grad():
            dice_score = L.dice_score_2d(registered_image, batch_fixed)
        
        return (
            total_loss.item(),
            dice_score.item(), 
            sim_loss.item(),
            (weights[1] * smooth_loss).item(),
            (weights[2] * keypoint_loss).item(),
            attention_loss.item()
        )

    def train_step_with_accumulation(self, batch_data, accumulation_step: int) -> Tuple[float, ...]:
        """
        Training step with gradient accumulation support for effective large batch training.
        
        Args:
            batch_data: Batch of training data
            accumulation_step: Current accumulation step
            
        Returns:
            Tuple[float, ...]: Training metrics
        """
        # Unpack batch data
        batch_fixed, batch_moving, fix_points, mov_points, mask_guide, gt_score_map, edge_map = batch_data
        
        # Move data to device
        batch_fixed = batch_fixed.to(self.device, non_blocking=True)
        batch_moving = batch_moving.to(self.device, non_blocking=True)
        fix_points = torch.round(fix_points).to(torch.int32).to(self.device, non_blocking=True)
        mov_points = mov_points.to(torch.float32).to(self.device, non_blocking=True)
        mask_guide = mask_guide.to(self.device, non_blocking=True)
        gt_score_map = gt_score_map.to(self.device, non_blocking=True)
        edge_map = edge_map.to(self.device, non_blocking=True) if edge_map is not None else None
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            if self.train_config["Em_map"]:
                if self.train_config["PAG_map"]:
                    registered_image, deformation_matrix, attention_score = self.model(
                        batch_moving, batch_fixed, edge_map
                    )
                    attention_loss = F.smooth_l1_loss(gt_score_map, attention_score, reduction="mean", beta=1.0)
                else:
                    registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, edge_map)
                    attention_loss = self.zero_tensor
            else:
                registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, mask_guide)
                attention_loss = self.zero_tensor
            
            # Compute losses
            sim_loss, smooth_loss = self._compute_losses(
                registered_image, batch_fixed, deformation_matrix, mask_guide
            )
            
            keypoint_loss = self._compute_keypoint_loss(deformation_matrix, fix_points, mov_points)
            
            # Compute total loss and apply accumulation
            weights = self.train_config["loss_weights"]
            # current_smooth_loss = smooth_loss.item()
            # weights = self._adjust_loss_weights(self.current_epoch, current_smooth_loss)
            
            total_loss = (weights[0] * sim_loss + 
                        weights[1] * smooth_loss + 
                        weights[2] * keypoint_loss)
            
            if self.train_config["PAG_map"]:
                total_loss += weights[3] * attention_loss
            
            # Gradient accumulation: divide by accumulation steps
            total_loss = total_loss / self.accumulation_steps
        
        # Backward pass
        self.scaler.scale(total_loss).backward()
        
        # Update parameters only when accumulation is complete
        if (accumulation_step + 1) % self.accumulation_steps == 0:
            # Gradient clipping for training stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # Parameter update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update global step and apply warmup
            self.current_step += 1
            if self.current_step <= self.warmup_steps:
                self._apply_warmup(self.current_step)
        
        # Compute metrics
        with torch.no_grad():
            dice_score = L.dice_score_2d(registered_image, batch_fixed)
        
        # Return actual loss values (not divided by accumulation steps, for display)
        return (
            (total_loss * self.accumulation_steps).item(),
            dice_score.item(), 
            sim_loss.item(),
            (weights[1] * smooth_loss).item(),
            (weights[2] * keypoint_loss).item(),
            attention_loss.item()
        )

    def validate_step(self, batch_data) -> Tuple[Tuple[float, ...], Tuple[float, float]]:
        """
        Execute one validation step - compute both loss metrics and distance metrics.
        
        Args:
            batch_data: Batch of validation data
            
        Returns:
            Tuple[Tuple[float, ...], Tuple[float, float]]: Loss metrics and distance metrics
        """
        batch_fixed, batch_moving, fix_points, mov_points, mask_guide, gt_score_map, edge_map = batch_data
        
        with torch.no_grad():
            # Move data to device
            batch_fixed = batch_fixed.to(self.device, non_blocking=True)
            batch_moving = batch_moving.to(self.device, non_blocking=True)
            fix_points = torch.round(fix_points).to(torch.int32).to(self.device, non_blocking=True)
            mov_points = mov_points.to(torch.float32).to(self.device, non_blocking=True)
            mask_guide = mask_guide.to(self.device, non_blocking=True)
            gt_score_map = gt_score_map.to(self.device, non_blocking=True) if gt_score_map is not None else None
            edge_map = edge_map.to(self.device, non_blocking=True) if edge_map is not None else None
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                # Forward pass
                if self.train_config["Em_map"]:
                    if self.train_config["PAG_map"]:
                        registered_image, deformation_matrix, attention_score = self.model(
                            batch_moving, batch_fixed, edge_map
                        )
                        if gt_score_map is not None:
                            attention_loss = F.smooth_l1_loss(gt_score_map, attention_score, reduction="mean", beta=1.0)
                        else:
                            attention_loss = torch.tensor(0.0, device=self.device)
                    else:
                        registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, edge_map)
                        attention_loss = torch.tensor(0.0, device=self.device)
                else:
                    registered_image, deformation_matrix = self.model(batch_moving, batch_fixed, mask_guide)
                    attention_loss = torch.tensor(0.0, device=self.device)
                
                # Compute loss metrics (same as training)
                sim_loss, smooth_loss = self._compute_losses(
                    registered_image, batch_fixed, deformation_matrix, mask_guide
                )
                
                keypoint_loss = self._compute_keypoint_loss(deformation_matrix, fix_points, mov_points)
                
                # Compute total loss
                weights = self.train_config["loss_weights"]
                total_loss = (weights[0] * sim_loss + 
                             weights[1] * smooth_loss + 
                             weights[2] * keypoint_loss)
                
                if self.train_config["PAG_map"]:
                    total_loss += weights[3] * attention_loss
                
                # Compute dice score
                dice_score = L.dice_score_2d(registered_image, batch_fixed)
            
            # Compute distance metrics (original validation) with sampling for efficiency
            if torch.rand(1).item() > 0.7:  # Only 30% of batches compute distance metrics
                avg_error, max_error = self._compute_distance_metrics(deformation_matrix, fix_points, mov_points)
            else:
                avg_error, max_error = 0.0, 0.0

            # Return both loss metrics and distance metrics
            loss_metrics = (
                total_loss.item(),
                dice_score.item(), 
                sim_loss.item(),
                (weights[1] * smooth_loss).item(),
                (weights[2] * keypoint_loss).item(),
                attention_loss.item()
            )
            
            distance_metrics = (avg_error, max_error)
            
            return loss_metrics, distance_metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one complete epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dict[str, float]: Averaged training metrics for the epoch
        """
        self.model.train()
        metrics = TrainingMetrics()
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {self.current_epoch + 1} - Training", 
            leave=False,
            unit="batch"
        )
        
        accumulation_step = 0
        for batch_idx, batch_data in enumerate(train_pbar):
            if self.accumulation_steps > 1:
                step_metrics = self.train_step_with_accumulation(batch_data, accumulation_step)
                accumulation_step += 1
            else:
                step_metrics = self.train_step(batch_data)
            
            metrics.update(step_metrics)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{step_metrics[0]:.4f}',
                'Dice': f'{step_metrics[1]:.4f}',
                'Sim': f'{step_metrics[2]:.4f}',
                'Keypoint': f'{step_metrics[4]:.4f}',
                'LR': f'{current_lr:.2e}',
                'Step': f'{self.current_step}'
            })

        averaged_metrics = metrics.average(len(train_loader))
    
        # Record training loss to history
        self.train_loss_history.append({
            'epoch': self.current_epoch,
            'metrics': averaged_metrics
        })
        
        return averaged_metrics

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one complete epoch - compute both loss and distance metrics.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dict[str, float]: Averaged validation metrics for the epoch
        """
        self.model.eval()
        val_metrics = ValidationMetrics()
        
        val_pbar = tqdm(
            val_loader, 
            desc=f"Epoch {self.current_epoch + 1} - Validation", 
            leave=False,
            unit="batch"
        )
        
        for batch_data in val_pbar:
            loss_metrics, distance_metrics = self.validate_step(batch_data)
            val_metrics.update(loss_metrics, distance_metrics)
            
            val_pbar.set_postfix({
                'Loss': f'{loss_metrics[0]:.4f}',
                'Dice': f'{loss_metrics[1]:.4f}',
                'Avg_Err': f'{distance_metrics[0]:.4f}',
                'Max_Err': f'{distance_metrics[1]:.4f}'
            })

        averaged_metrics = val_metrics.average(len(val_loader))
    
        self.val_loss_history.append({
            'epoch': self.current_epoch,
            'metrics': averaged_metrics
        })
        
        return averaged_metrics

    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, log_writer: SummaryWriter):
        """Main training loop"""
        logging.info(f"Starting training for {num_epochs} epochs")
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            start_time = time.time()
            
            # Update learning rate
            if hasattr(self.lr_scheduler, 'step') and not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if not isinstance(self.lr_scheduler, LearningRateScheduler):
                    self.lr_scheduler.step()
                else:
                    self.lr_scheduler.step(epoch)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Logging
            epoch_time = (time.time() - start_time) / 60
            
            # Log training metrics
            for metric_name, value in train_metrics.items():
                log_writer.add_scalar(f"train/{metric_name}", value, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            log_writer.add_scalar("learning_rate", current_lr, epoch)
            log_writer.add_scalar("scaler_scale", self.scaler.get_scale(), epoch)
            
            # Save model checkpoint
            self.model_saver.save_checkpoint(
                self.model.state_dict(), 
                self.optimizer.state_dict(),
                self.scaler.state_dict(),
                epoch
            )
            
            # Validation phase (every 5 epochs)
            val_metrics = {}
            if (epoch + 1) % 10 == 0:
                val_metrics = self.validate_epoch(val_loader)

                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics['total_loss'])
                
                # Log validation metrics
                for metric_name, value in val_metrics.items():
                    log_writer.add_scalar(f"val/{metric_name}", value, epoch)
                
                # Save best model (both loss and error based)
                self.model_saver.save_best(
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    self.scaler.state_dict(),
                    val_metrics['total_loss'], 
                    val_metrics['avg_error'],
                    epoch
                )
            
            # Update progress bar and log
            progress_info = {
                'Loss': f'{train_metrics["total_loss"]:.4f}',
                'Dice': f'{train_metrics["dice_score"]:.4f}',
                'LR': f'{current_lr:.2e}'
            }
            
            if val_metrics:
                progress_info['Val_Loss'] = f'{val_metrics["total_loss"]:.4f}'
                progress_info['Val_Err'] = f'{val_metrics["avg_error"]:.4f}'
            
            epoch_pbar.set_postfix(progress_info)
            
            # Detailed logging
            log_msg = f'[{epoch_time:.2f}min] Epoch {epoch + 1}/{num_epochs}:'
            log_msg += f'\n  📊 Train - Loss: {train_metrics["total_loss"]:.6f}, Dice: {train_metrics["dice_score"]:.6f}'
            log_msg += f'\n  🔧 Components - Sim: {train_metrics["sim_loss"]:.6f}, Smooth: {train_metrics["smooth_loss"]:.6f}, Keypoint: {train_metrics["keypoint_loss"]:.6f}'
            
            if val_metrics:
                log_msg += f'\n  ✅ Val Loss - Total: {val_metrics["total_loss"]:.6f}, Dice: {val_metrics["dice_score"]:.6f}'
                log_msg += f'\n  📏 Val Distance - Avg Error: {val_metrics["avg_error"]:.6f}, Max Error: {val_metrics["max_error"]:.6f}'
            
            tqdm.write(log_msg)
            tqdm.write("-" * 80)

            # Save the loss history every 10 epochs.
            if (epoch + 1) % 10 == 0:
                self._save_loss_history()
                
            # Memory cleanup
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()


        self._save_loss_history()
        
        epoch_pbar.close()
        logging.info("🎉 Training completed successfully!")
        logging.info(f"📈 Best validation loss: {self.model_saver.best_loss_score:.6f}")
        logging.info(f"📏 Best validation error: {self.model_saver.best_error_score:.6f}")