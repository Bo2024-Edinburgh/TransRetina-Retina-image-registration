import os  
import logging  
from typing import Tuple, Dict, List 

import torch  
import torch.nn as nn  


def print_model_architecture(model: nn.Module, input_shape: Tuple[int, ...], device: torch.device):
    """
    Print detailed model architecture information.
    
    Args:
        model (nn.Module): PyTorch model to inspect.
        input_shape (Tuple[int, ...]): Input tensor shape (C, H, W).
        device (torch.device): Device to use for dummy input and forward pass.
    """
    print("\n" + "="*80)
    print("🏗️  MODEL ARCHITECTURE")
    print("="*80)

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📋 Model Type: {model.__class__.__name__}")
    print(f"📏 Input Shape: {input_shape}")
    print(f"🔢 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\n" + "-"*80)
    print("🔍 DETAILED ARCHITECTURE")
    print("-"*80)
    print(model)

    print("\n" + "-"*80)
    print("⚙️  COMPONENT BREAKDOWN")
    print("-"*80)

    # Count parameters per submodule
    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params
        print(f"  {name:20s}: {params:>10,} parameters")

    print("\n" + "-"*80)
    print("🔄 FORWARD PASS TEST")
    print("-"*80)

    model.eval()
    with torch.no_grad():
        try:
            dummy_input_moving = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).to(device)
            dummy_input_fixed = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).to(device)
            dummy_edge_map = torch.randn(1, input_shape[1], input_shape[2]).to(device)

            print(f"  Input Moving Shape: {dummy_input_moving.shape}")
            print(f"  Input Fixed Shape:  {dummy_input_fixed.shape}")
            print(f"  Input Edge Map:     {dummy_edge_map.shape}")

            outputs = model(dummy_input_moving, dummy_input_fixed, dummy_edge_map)

            # Handle different output formats
            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    registered_img, deformation_field = outputs
                    print(f"  Output Registered:  {registered_img.shape}")
                    print(f"  Output Deformation: {deformation_field.shape}")
                elif len(outputs) == 3:
                    registered_img, deformation_field, attention_score = outputs
                    print(f"  Output Registered:  {registered_img.shape}")
                    print(f"  Output Deformation: {deformation_field.shape}")
                    print(f"  Output Attention:   {attention_score.shape}")
            else:
                print(f"  Output Shape: {outputs.shape}")

            print("  ✅ Forward pass successful!")

        except Exception as e:
            print(f"  ❌ Forward pass failed: {str(e)}")

    print("\n" + "="*80)
    print("🏗️  MODEL ARCHITECTURE SUMMARY COMPLETE")
    print("="*80 + "\n")


def count_parameters_by_layer_type(model: nn.Module) -> Dict[str, int]:
    """Count total number of parameters for each layer type in the model."""
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        if layer_type not in layer_counts:
            layer_counts[layer_type] = 0
        params = sum(p.numel() for p in module.parameters(recurse=False))
        layer_counts[layer_type] += params
    return {k: v for k, v in layer_counts.items() if v > 0}


def print_layer_statistics(model: nn.Module):
    """Print number and percentage of parameters by layer type."""
    layer_counts = count_parameters_by_layer_type(model)
    print("\n" + "-"*60)
    print("📊 PARAMETER STATISTICS BY LAYER TYPE")
    print("-"*60)
    total_params = sum(layer_counts.values())
    for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"  {layer_type:20s}: {count:>10,} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_params:>10,} (100.0%)")


class ModelSaver:
    """Handles model checkpointing, directory creation, and best model tracking."""

    def __init__(self, save_dir: str, model_prefix: str, save_interval: int = 10):
        self.save_dir = Path(save_dir)
        self.model_prefix = model_prefix
        self.save_interval = save_interval
        self.best_loss_score = float('inf')
        self.best_error_score = float('inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not os.access(str(self.save_dir), os.W_OK):
            raise PermissionError(f"No write permission to {self.save_dir}")

        logging.info(f"📁 Save directory created: {self.save_dir}")

    def save_checkpoint(self, model_state: dict, optimizer_state: dict, scaler_state: dict,
                        epoch: int, force: bool = False):
        """Save model checkpoint at specified interval or when forced."""
        print(f"DEBUG: Attempting to save checkpoint for epoch {epoch}")
        print(f"DEBUG: Save directory: {self.save_dir}")
        print(f"DEBUG: Directory exists: {self.save_dir.exists()}")
        print(f"DEBUG: Directory writable: {os.access(str(self.save_dir), os.W_OK)}")

        if epoch % self.save_interval == 0 or force:
            checkpoint_path = self.save_dir / f"{self.model_prefix}_epoch_{epoch}.pth"
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'scaler_state_dict': scaler_state,
                'epoch': epoch
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"✓ Saved checkpoint: {checkpoint_path}")

    def save_best(self, model_state: dict, optimizer_state: dict, scaler_state: dict,
                  loss_score: float, error_score: float, epoch: int) -> Tuple[bool, bool]:
        """Save model if it achieves a new best loss or error score."""
        saved_loss = False
        saved_error = False

        if loss_score < self.best_loss_score:
            self.best_loss_score = loss_score
            best_loss_path = self.save_dir / f"{self.model_prefix}_best_loss.pth"
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'scaler_state_dict': scaler_state,
                'epoch': epoch,
                'best_loss': loss_score,
                'best_error': error_score
            }
            torch.save(checkpoint, best_loss_path)
            logging.info(f"✓ New best loss model saved! Loss: {loss_score:.6f} at epoch {epoch}")
            saved_loss = True

        if error_score < self.best_error_score:
            self.best_error_score = error_score
            best_error_path = self.save_dir / f"{self.model_prefix}_best_error.pth"
            checkpoint = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'scaler_state_dict': scaler_state,
                'epoch': epoch,
                'best_loss': loss_score,
                'best_error': error_score
            }
            torch.save(checkpoint, best_error_path)
            logging.info(f"✓ New best error model saved! Error: {error_score:.6f} at epoch {epoch}")
            saved_error = True

        return saved_loss, saved_error


class LearningRateScheduler:
    """Custom learning rate scheduler with predefined decay steps."""

    def __init__(self, optimizer, initial_lr: float, decay_epochs: List[int], decay_factor: float = 0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor
        self.current_lr = initial_lr

    def step(self, epoch: int):
        """Adjust learning rate at specific epochs."""
        if epoch in self.decay_epochs:
            decay_count = self.decay_epochs.index(epoch) + 1
            self.current_lr = self.initial_lr * (self.decay_factor ** decay_count)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr

            logging.info(f"📉 Learning rate updated to: {self.current_lr:.2e}")
