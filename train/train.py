#!/usr/bin/env python3
"""
Optimized training script for CasReg model with dHCP data.

This script implements medical image registration training using cascaded networks
with Neo-Hookean regularization and entropy-based ratio learning.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to fix tkinter errors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from models import losses
from utils.training_utils import AverageMeter, save_checkpoint, Logger, lr_decay
from utils.metrics import dice_val
from utils.spatial import register_model
from utils.math_ops import jacobian_determinant_vxm
from data import datasets
from models.models import CasReg_nhe

# Configuration constants
DEFAULT_CONFIG = {
    'IMG_SIZE': (192, 256, 256),
    'MAX_STEPS_TRAIN': 100,
    'MAX_STEPS_VAL': 20,
    'SAVE_RATIO_INTERVAL': 10,
    'DEFAULT_EPOCHS': 500,
    'DEFAULT_LR': 1e-3,
    'DEFAULT_LAMBDA': 0.01,
    'DEFAULT_NB_CASCADES': 5,
    'DEFAULT_NB_LABELS': 8,
    'WEIGHT_DECAY': 0.0,
}


class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.img_size = tuple(args.img_size) if hasattr(args, 'img_size') else DEFAULT_CONFIG['IMG_SIZE']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training parameters
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.lambda_reg = args.lambda_
        self.weight_entropy = args.weight_entropy

        # Model parameters
        self.nb_cascades = args.nb_cascades
        self.coef_ratio = args.coef_ratio
        self.mode = args.mode

        # Paths
        self.save_dir = Path(args.save_dir)
        self.weights_dir = self.save_dir / "weights"
        self.loss_dir = self.save_dir / "loss"

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for saving outputs."""
        for directory in [self.save_dir, self.weights_dir, self.loss_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class TrainingManager:
    """Manages the complete training process for CasReg model."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.writer = SummaryWriter(log_dir=str(config.save_dir / "tensorboard"))

        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.reg_models = {}

        # Training state
        self.current_epoch = 0
        self.best_score = 0.0

        # Fixed visualization pair (same images for all visualizations)
        self.viz_moving = None
        self.viz_fixed = None
        self.viz_fixed_seg = None
        self.viz_gw1 = None
        self.viz_gw2 = None
        self.viz_filename = None

        # Persistent data generator (reuse instead of creating every iteration)
        self.data_generator = None

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def setup_model(self) -> None:
        """Initialize and configure the model."""
        try:
            self.model = CasReg_nhe(
                self.config.img_size,
                batch_size=1,
                coef_ratio=self.config.coef_ratio,
                n_casc=self.config.nb_cascades,
                mode=self.config.mode,
                layer_mode="contracted"
            )

            # Load pretrained weights if provided
            if hasattr(self.config.args, 'in_weights') and self.config.args.in_weights:
                self._load_pretrained_weights(self.config.args.in_weights)

            self.model.to(self.config.device)

            # Initialize ratio parameters if requested
            if getattr(self.config.args, 'reinit_ratio', 0) == 1:
                self._initialize_ratio_parameters()

            self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

        except Exception as e:
            self.logger.error(f"Failed to setup model: {e}")
            raise

    def _load_pretrained_weights(self, weights_path: str) -> None:
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(weights_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained weights: {e}")
            raise

    def _initialize_ratio_parameters(self) -> None:
        """Initialize ratio parameters with normal distribution."""
        if hasattr(self.model, 'ratio'):
            nn.init.normal_(self.model.ratio.weight, mean=0, std=1e-5)
            nn.init.constant_(self.model.ratio.bias, 0)
            self.logger.info("Ratio parameters reinitialized")

    def setup_visualization_pair(self) -> None:
        """Setup a fixed pair of images for consistent visualization across training."""
        try:
            # Generate one fixed pair for all visualizations
            data_generator = datasets.volgen_dhcp_hybrid_inv(self.config.args.npz_dhcp)
            self.viz_moving, self.viz_fixed, self.viz_fixed_seg, self.viz_gw1, self.viz_gw2, self.viz_filename = next(data_generator)

            self.logger.info(f"Fixed visualization pair: {self.viz_filename} (GW1: {self.viz_gw1:.1f}, GW2: {self.viz_gw2:.1f})")

        except Exception as e:
            self.logger.error(f"Failed to setup visualization pair: {e}")
            raise

    def setup_training_components(self) -> None:
        """Setup optimizer, loss function, and registration models."""
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=DEFAULT_CONFIG['WEIGHT_DECAY'],
            amsgrad=True
        )

        # Setup loss function
        use_entropy = getattr(self.config.args, 'entropy', 1) == 1
        normalize = getattr(self.config.args, 'norm_loss', 1) == 1

        self.loss_function = losses.local_grad_neo_hookean_dHCP(
            shape=self.config.img_size,
            return_maps=False,
            use_entropy=use_entropy,
            weight_entropy=self.config.weight_entropy,
            normalize=normalize
        )

        # Setup registration models
        self.reg_models = {
            'nearest': register_model(self.config.img_size, 'nearest').to(self.config.device),
            'bilinear': register_model(self.config.img_size, 'bilinear').to(self.config.device)
        }

        # OPTIMIZATION: Create data generator ONCE instead of every iteration
        self.logger.info("Setting up persistent data generator...")
        self.data_generator = datasets.volgen_dhcp_hybrid_inv(self.config.args.npz_dhcp)

        self.logger.info("Training components initialized successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()

        metrics = {
            'total_loss': 0.0,
            'ncc_loss': 0.0,
            'nhe_loss': 0.0,
            'entropy_loss': 0.0,
        }

        # Update learning rate
        current_lr = lr_decay(self.config.learning_rate, epoch, self.config.epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        for step in range(1, DEFAULT_CONFIG['MAX_STEPS_TRAIN'] + 1):
            try:
                # OPTIMIZATION: Use persistent data generator (no recreation every iteration!)
                moving, fixed, fixed_seg, gw1, gw2, filename = next(self.data_generator)

                # Forward pass
                input_tensor = torch.cat((moving, fixed), dim=1)

                # Forward pass
                output = self.model(input_tensor)

                # Calculate losses
                ncc_loss = losses.NCC().forward(output[0], fixed)
                nhe_loss, ratio_cpu, entropy_loss = self.loss_function.forward(
                    output[1], output[-1], fixed_seg
                )

                print("weight reg :", self.config.lambda_reg)
                print("weight ent :", self.config.weight_entropy)
                total_loss = ncc_loss + self.config.lambda_reg * nhe_loss + self.config.weight_entropy * entropy_loss

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Clear intermediate tensors to free memory
                del input_tensor

                # Update metrics
                metrics['total_loss'] += total_loss.item()
                metrics['ncc_loss'] += ncc_loss.item()
                metrics['nhe_loss'] += nhe_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()

                # Analyze deformation field (optimize memory)
                with torch.no_grad():
                    # Move to CPU immediately and calculate stats efficiently
                    def_field_cpu = output[1][0].cpu()  # First batch element only
                    def_magnitude = torch.mean(torch.sqrt(torch.sum(def_field_cpu**2, dim=0))).item()

                    # Calculate Jacobian determinant every iteration
                    jac_det = jacobian_determinant_vxm(def_field_cpu.numpy())
                    avg_jac_det = np.mean(jac_det)
                    neg_jac_percent = 100.0 * np.sum(jac_det <= 0) / jac_det.size

                    # Clear GPU memory
                    del def_field_cpu
                    torch.cuda.empty_cache()

                # ============ ORGANIZED OUTPUT ============
                print(f"\n=== Iter {step}/{DEFAULT_CONFIG['MAX_STEPS_TRAIN']} ===")
                print(f"Subject: {filename} (GW1: {gw1:.1f}, GW2: {gw2:.1f})")

                # Loss information
                print(f"Loss: {total_loss.item():.4f} | Sim: {ncc_loss.item():.4f}, Reg: {nhe_loss.item():.6f}, Ent: {entropy_loss.item():.6f} | LR: {current_lr:.6f}")

                # Deformation field analysis
                print(f"Deformation: |u|_avg: {def_magnitude:.3f}mm | Jac: {avg_jac_det:.3f}±{np.std(jac_det):.3f}, neg: {neg_jac_percent:.1f}%")

                # Biomechanical parameters (optimized)
                if isinstance(ratio_cpu, tuple):
                    if len(ratio_cpu) == 5:
                        # Neo-Hookean with entropy: (lambda_cpu, mu_cpu, mask_grad, stretch, vol_change)
                        lambda_cpu, mu_cpu = ratio_cpu[0], ratio_cpu[1]  # Only extract what we need
                        print(f"Material: κ: {np.mean(lambda_cpu):.4f}±{np.std(lambda_cpu):.4f}, μ: {np.mean(mu_cpu):.4f}±{np.std(mu_cpu):.4f}")
                    elif len(ratio_cpu) == 2:
                        # Simple Neo-Hookean: (mu_cpu, lambda_cpu)
                        mu_cpu, lambda_cpu = ratio_cpu
                        print(f"Material: μ: {np.mean(mu_cpu):.4f}±{np.std(mu_cpu):.4f}, κ: {np.mean(lambda_cpu):.4f}±{np.std(lambda_cpu):.4f}")
                    else:
                        print(f"Material: {len(ratio_cpu)} parameters available")

                print()  # Blank line for readability

                # Save ratio and visualization periodically
                if step % DEFAULT_CONFIG['SAVE_RATIO_INTERVAL'] == 0:
                    np.save(self.config.save_dir / "mu.npy", ratio_cpu)  # Use original filename

                    # Generate outputs for the FIXED visualization pair (memory optimized)
                    self.model.eval()  # Set to eval mode for visualization
                    with torch.no_grad():
                        # Clear cache before visualization
                        torch.cuda.empty_cache()

                        viz_input = torch.cat((self.viz_moving, self.viz_fixed), dim=1)
                        viz_output = self.model(viz_input)

                        # Move to CPU immediately to free GPU memory
                        viz_warped = viz_output[0].cpu()
                        viz_deformation = viz_output[1].cpu().numpy()

                        # Compute material parameters
                        _, viz_ratio_cpu, _ = self.loss_function.forward(
                            viz_output[1], viz_output[-1], self.viz_fixed_seg
                        )

                        # Clear GPU tensors
                        del viz_input, viz_output
                        torch.cuda.empty_cache()

                        # Compute Jacobian on CPU
                        viz_jac_det = jacobian_determinant_vxm(viz_deformation[0])

                    # Save comprehensive analysis figure using FIXED pair
                    self.save_visualization(
                        step=step,
                        moving=self.viz_moving.cpu(),  # Ensure CPU tensors for visualization
                        fixed=self.viz_fixed.cpu(),
                        warped=viz_warped,
                        deformation_field=viz_deformation,
                        ratio_cpu=viz_ratio_cpu,
                        jac_det=viz_jac_det,
                        filename=self.viz_filename
                    )

                    # Clear visualization memory and return to training mode
                    del viz_warped, viz_deformation, viz_jac_det
                    torch.cuda.empty_cache()
                    self.model.train()  # Return to training mode

            except Exception as e:
                import traceback
                self.logger.error(f"Error in training step {step}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Clear GPU memory on error to prevent accumulation
                torch.cuda.empty_cache()
                continue

        # Clear GPU cache at end of training epoch
        torch.cuda.empty_cache()

        # Average the metrics
        num_steps = DEFAULT_CONFIG['MAX_STEPS_TRAIN']
        for key in metrics:
            metrics[key] /= num_steps

        return metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute validation for one epoch."""
        self.model.eval()

        total_ncc = 0.0

        with torch.no_grad():
            for step in range(1, DEFAULT_CONFIG['MAX_STEPS_VAL'] + 1):
                try:
                    # OPTIMIZATION: Use persistent data generator (no recreation!)
                    moving, fixed, fixed_seg, _, _, _ = next(self.data_generator)  # Ignore gw1, gw2, filename in validation
                    input_tensor = torch.cat((moving, fixed), dim=1)
                    output = self.model(input_tensor)

                    ncc_loss = losses.NCC().forward(output[0], fixed)
                    total_ncc += -ncc_loss.detach().cpu().numpy()

                except Exception as e:
                    import traceback
                    self.logger.error(f"Error in validation step {step}: {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

        avg_ncc = total_ncc / DEFAULT_CONFIG['MAX_STEPS_VAL']
        self.logger.info(f'Validation - Epoch {epoch}, Average NCC: {avg_ncc:.6f}')

        return {'avg_ncc': avg_ncc}

    def save_model_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with training metrics."""
        try:
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_score': self.best_score,
                'nb_cascades': self.config.nb_cascades,
                'mode': self.config.mode,
                'metrics': metrics
            }

            filename = f'ncc{metrics.get("avg_ncc", 0.0):.3f}_epoch_{epoch}.pth.tar'
            save_checkpoint(
                checkpoint,
                save_dir=str(self.config.weights_dir) + "/",
                filename=filename
            )

            self.logger.info(f"Model checkpoint saved: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def save_visualization(self, step: int, moving: torch.Tensor, fixed: torch.Tensor,
                         warped: torch.Tensor, deformation_field: np.ndarray,
                         ratio_cpu: tuple, jac_det: np.ndarray, filename: str) -> None:
        """Save comprehensive biomechanical analysis visualization."""
        try:
            # Get middle slice for visualization (axial view)
            mid_slice = deformation_field.shape[-1] // 2

            # Convert tensors to numpy for visualization (memory efficient)
            with torch.no_grad():
                moving_np = moving[0, 0, :, :, mid_slice].numpy() if moving.is_cpu else moving[0, 0, :, :, mid_slice].cpu().numpy()
                fixed_np = fixed[0, 0, :, :, mid_slice].numpy() if fixed.is_cpu else fixed[0, 0, :, :, mid_slice].cpu().numpy()
                warped_np = warped[0, 0, :, :, mid_slice].numpy() if warped.is_cpu else warped[0, 0, :, :, mid_slice].cpu().numpy()

            # Create figure with subplots (smaller for memory)
            fig = plt.figure(figsize=(16, 10), dpi=100)  # Reduced size and DPI
            gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

            # Calculate deformation magnitude (local modulus)
            def_mag = np.sqrt(np.sum(deformation_field[0]**2, axis=0))[:, :, mid_slice]

            # Extract material parameters
            if isinstance(ratio_cpu, tuple) and len(ratio_cpu) >= 2:
                if len(ratio_cpu) == 5:
                    # (lambda_cpu, mu_cpu, mask_grad, stretch, vol_change)
                    lambda_cpu, mu_cpu = ratio_cpu[0], ratio_cpu[1]
                else:
                    # (mu_cpu, lambda_cpu)
                    mu_cpu, lambda_cpu = ratio_cpu[0], ratio_cpu[1]

                kappa_slice = lambda_cpu[:, :, mid_slice]
                mu_slice = mu_cpu[:, :, mid_slice]
            else:
                kappa_slice = np.zeros_like(def_mag)
                mu_slice = np.zeros_like(def_mag)

            jac_slice = jac_det[:, :, mid_slice]

            # Plot images
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(moving_np, cmap='gray', aspect='equal')
            ax1.set_title('Moving Image', fontsize=12, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.8)

            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(fixed_np, cmap='gray', aspect='equal')
            ax2.set_title('Fixed Image', fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, shrink=0.8)

            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(warped_np, cmap='gray', aspect='equal')
            ax3.set_title('Warped Image', fontsize=12, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, shrink=0.8)

            ax4 = fig.add_subplot(gs[0, 3])
            im4 = ax4.imshow(def_mag, cmap='viridis', aspect='equal')
            ax4.set_title(f'Deformation |u| (mm)\nAvg: {np.mean(def_mag):.2f}±{np.std(def_mag):.2f}',
                         fontsize=12, fontweight='bold')
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, shrink=0.8)

            # Plot material parameters and Jacobian
            ax5 = fig.add_subplot(gs[1, 0])
            im5 = ax5.imshow(mu_slice, cmap='plasma', aspect='equal')
            ax5.set_title(f'Shear Modulus μ\nAvg: {np.mean(mu_slice):.3f}±{np.std(mu_slice):.3f}',
                         fontsize=12, fontweight='bold')
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, shrink=0.8)

            ax6 = fig.add_subplot(gs[1, 1])
            im6 = ax6.imshow(kappa_slice, cmap='inferno', aspect='equal')
            ax6.set_title(f'Bulk Modulus κ\nAvg: {np.mean(kappa_slice):.3f}±{np.std(kappa_slice):.3f}',
                         fontsize=12, fontweight='bold')
            ax6.axis('off')
            plt.colorbar(im6, ax=ax6, shrink=0.8)

            ax7 = fig.add_subplot(gs[1, 2])
            im7 = ax7.imshow(jac_slice, cmap='RdBu_r', vmin=0.5, vmax=1.5, aspect='equal')
            ax7.set_title(f'Jacobian Determinant\nAvg: {np.mean(jac_slice):.3f}, Neg: {100*np.sum(jac_slice<=0)/jac_slice.size:.1f}%',
                         fontsize=12, fontweight='bold')
            ax7.axis('off')
            plt.colorbar(im7, ax=ax7, shrink=0.8)

            # Difference image
            ax8 = fig.add_subplot(gs[1, 3])
            diff_img = np.abs(warped_np - fixed_np)
            im8 = ax8.imshow(diff_img, cmap='hot', aspect='equal')
            ax8.set_title(f'|Warped - Fixed|\nMSE: {np.mean(diff_img**2):.4f}',
                         fontsize=12, fontweight='bold')
            ax8.axis('off')
            plt.colorbar(im8, ax=ax8, shrink=0.8)

            # Add main title
            fig.suptitle(f'Biomechanical Analysis - Step {step} - {filename} [FIXED PAIR]\nSlice: {mid_slice}/{deformation_field.shape[-1]}',
                        fontsize=16, fontweight='bold', y=0.95)

            # Save figure (memory optimized)
            fig_path = self.config.save_dir / f"analysis_step_{step:03d}.png"
            plt.savefig(fig_path, dpi=100, bbox_inches='tight', format='png',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            plt.clf()  # Clear the figure completely

            # Clear matplotlib memory
            fig.clear()
            del fig

            self.logger.info(f"Saved analysis figure: {fig_path}")

        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")
            # Ensure figure is closed even on error
            plt.close('all')

    def log_training_metrics(self, epoch: int, train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float]) -> None:
        """Log training metrics to tensorboard and file."""
        # Tensorboard logging
        self.writer.add_scalar("Loss/Train_Total", train_metrics['total_loss'], epoch)
        self.writer.add_scalar("Loss/Train_NCC", train_metrics['ncc_loss'], epoch)
        self.writer.add_scalar("Loss/Train_NHE", train_metrics['nhe_loss'], epoch)
        self.writer.add_scalar("Loss/Train_Entropy", train_metrics['entropy_loss'], epoch)
        self.writer.add_scalar("Validation/NCC", val_metrics['avg_ncc'], epoch)

        # File logging
        try:
            loss_file = self.config.loss_dir / "loss.txt"
            with open(loss_file, 'a') as f:
                f.write(f"\n{epoch} {train_metrics['total_loss']:.6f} {val_metrics['avg_ncc']:.6f}")
        except Exception as e:
            self.logger.error(f"Failed to write loss file: {e}")

    def train(self) -> None:
        """Execute the complete training process."""
        self.logger.info("Starting training process...")
        self.logger.info(f"Training configuration: {vars(self.config.args)}")

        try:
            for epoch in range(1, self.config.epochs + 1):
                self.current_epoch = epoch

                # Training phase
                train_metrics = self.train_epoch(epoch)

                # Validation phase
                val_metrics = self.validate_epoch(epoch)

                # Update best score
                current_score = val_metrics['avg_ncc']
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.logger.info(f"New best validation score: {self.best_score:.6f}")

                # Save checkpoint
                all_metrics = {**train_metrics, **val_metrics}
                self.save_model_checkpoint(epoch, all_metrics)

                # Log metrics
                self.log_training_metrics(epoch, train_metrics, val_metrics)

                self.logger.info(f"Epoch {epoch} completed successfully")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.writer.close()
            self.logger.info("Training process finished")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with improved organization."""
    parser = argparse.ArgumentParser(
        description="Training script for ViT-VNet 3D fetal brain MRI registration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--npz_dhcp", type=str, required=True,
                         help="Path to preprocessed dHCP npz files directory")
    io_group.add_argument("--save_dir", type=str, required=True,
                         help="Directory to save training outputs")
    io_group.add_argument("--in_weights", type=str, default=None,
                         help="Path to pretrained model weights")

    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--nb_cascades", type=int, default=DEFAULT_CONFIG['DEFAULT_NB_CASCADES'],
                           help="Number of cascaded networks")
    model_group.add_argument("--mode", type=str, default="cascade",
                           choices=["cascade", "parallel"],
                           help="Training mode: cascade or parallel")
    model_group.add_argument("--img_size", nargs=3, type=int, default=DEFAULT_CONFIG['IMG_SIZE'],
                           help="Image dimensions (D H W)")
    model_group.add_argument("--nb_labels", type=int, default=DEFAULT_CONFIG['DEFAULT_NB_LABELS'],
                           help="Number of segmentation labels")

    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument("--epochs", type=int, default=DEFAULT_CONFIG['DEFAULT_EPOCHS'],
                           help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=DEFAULT_CONFIG['DEFAULT_LR'],
                           help="Initial learning rate")
    train_group.add_argument("--lambda_", type=float, default=DEFAULT_CONFIG['DEFAULT_LAMBDA'],
                           help="Regularization weight")
    train_group.add_argument("--dice_weight", type=float, default=0.01,
                           help="Dice loss weight (for supervised training)")

    # Regularization parameters
    reg_group = parser.add_argument_group('Regularization')
    reg_group.add_argument("--regularization", type=str, default="NHE",
                         choices=["grad", "NHE"],
                         help="Regularization type")
    reg_group.add_argument("--ratio_nhe", type=float, default=5.0,
                         help="Ratio between shear and bulk modulus")
    reg_group.add_argument("--entropy", type=int, default=1, choices=[0, 1],
                         help="Use variable ratio with entropy loss")
    reg_group.add_argument("--coef_ratio", type=float, default=10.0,
                         help="Ratio coefficient")
    reg_group.add_argument("--weight_entropy", type=float, default=1e-5,
                         help="Entropy loss weight")

    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument("--supervised", type=int, default=0, choices=[0, 1],
                              help="Supervised (1) or unsupervised (0) training")
    advanced_group.add_argument("--contracted", type=int, default=1, choices=[0, 1],
                              help="Use contracted architecture")
    advanced_group.add_argument("--with_maps", type=int, default=0, choices=[0, 1],
                              help="Include regularization maps")
    advanced_group.add_argument("--reinit_ratio", type=int, default=0, choices=[0, 1],
                              help="Reinitialize ratio parameters")
    advanced_group.add_argument("--mask_grad", type=int, default=1, choices=[0, 1],
                              help="Use gradient masking")
    advanced_group.add_argument("--norm_loss", type=int, default=1, choices=[0, 1],
                              help="Normalize loss function")

    return parser.parse_args()


def main():
    """Main training function."""
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Parse arguments and create configuration
    args = parse_arguments()
    config = TrainingConfig(args)

    # Setup training manager
    trainer = TrainingManager(config)

    # Initialize components
    trainer.setup_model()
    trainer.setup_training_components()
    trainer.setup_visualization_pair()

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()