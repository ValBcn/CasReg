#!/usr/bin/env python3
"""
Unified Multi-Atlas Segmentation and Registration Script
Refactored from mas/ directory - combines all registration workflows

Supported modes:
- basic: Basic image registration
- biomechanical: Registration with biomechanical constraints
- multi_atlas: Multi-atlas segmentation
- visualization: Create plots and grids
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path

# Add project directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))
sys.path.append(os.path.join(PROJECT_ROOT, 'models'))

from utils.registration_utils import *
from utils.io_utils import resize_and_uncrop
from models import losses
from torch import optim
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Multi-Atlas Segmentation and Registration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic registration
  python mas_unified.py --mode basic --moving /path/to/moving.npz --fixed /path/to/fixed.npz --weights /path/to/model.pth

  # Multi-atlas segmentation
  python mas_unified.py --mode multi_atlas --atlas_dir /path/to/atlases --target /path/to/target.npz --weights /path/to/model.pth

  # Biomechanical registration
  python mas_unified.py --mode biomechanical --moving /path/to/moving.npz --fixed /path/to/fixed.npz --weights /path/to/biomech_model.pth
        """
    )

    # Core arguments
    parser.add_argument("--mode", type=str, required=True,
                        choices=['basic', 'biomechanical', 'multi_atlas', 'visualization'],
                        help="Registration mode")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")

    # Data arguments
    parser.add_argument("--moving", type=str,
                        help="Path to moving image (NPZ file)")

    parser.add_argument("--fixed", type=str,
                        help="Path to fixed image (NPZ file)")

    parser.add_argument("--target", type=str,
                        help="Path to target image for multi-atlas mode")

    parser.add_argument("--atlas_dir", type=str,
                        help="Directory containing atlas NPZ files")

    # Model configuration
    parser.add_argument("--model_type", type=str, default="casreg",
                        choices=['casreg', 'casreg_nhe'],
                        help="Model architecture type")

    parser.add_argument("--imgsize", nargs=3, type=int, default=[192, 256, 256],
                        help="Image dimensions (D H W)")

    # Registration parameters
    parser.add_argument("--n_atlases", type=int, default=5,
                        help="Number of atlases to use for multi-atlas segmentation")

    parser.add_argument("--fusion_method", type=str, default="majority",
                        choices=['majority', 'weighted', 'best'],
                        help="Label fusion method for multi-atlas")

    parser.add_argument("--nb_labels", type=int, default=8,
                        help="Number of segmentation labels")

    # Fine-tuning parameters
    parser.add_argument("--finetune_iterations", type=int, default=0,
                        help="Number of fine-tuning iterations (0 = no fine-tuning)")

    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for fine-tuning")

    # Postprocessing
    parser.add_argument("--smoothing", type=str, default="none",
                        choices=['none', 'median', 'gaussian'],
                        help="Segmentation smoothing method")

    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Kernel size for smoothing")

    # Visualization
    parser.add_argument("--save_plots", action='store_true',
                        help="Save visualization plots")

    parser.add_argument("--plot_slice", type=int, default=None,
                        help="Slice index for 2D visualization (default: middle)")

    # Advanced options
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (-1 for CPU)")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")

    parser.add_argument("--verbose", action='store_true',
                        help="Verbose output")

    return parser.parse_args()


class BasicRegistrationWorkflow(RegistrationWorkflow):
    """Basic pairwise image registration"""

    def run_registration(self, moving_data, fixed_data):
        """Run basic registration between two images"""

        # Prepare input tensors
        x_in, _, moving_tensor, fixed_tensor = prepare_data_for_registration(
            moving_data, fixed_data, channels=['vol'])

        if x_in is None:
            # Fallback to vol1/vol2 if vol not available - try different combinations
            if 'vol1' in moving_data and 'vol2' in fixed_data:
                # Create temporary data with consistent naming
                moving_temp = moving_data.copy()
                fixed_temp = fixed_data.copy()
                if 'vol1' in moving_temp:
                    moving_temp['vol'] = moving_temp['vol1']
                if 'vol2' in fixed_temp:
                    fixed_temp['vol'] = fixed_temp['vol2']
                x_in, _, moving_tensor, fixed_tensor = prepare_data_for_registration(
                    moving_temp, fixed_temp, channels=['vol'])

        # Run registration
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, '__call__'):
                output = self.model(x_in)
                if isinstance(output, (list, tuple)):
                    warped, flow = output[0], output[1]
                    flow_list = output[2] if len(output) > 2 else None
                else:
                    warped, flow, flow_list = output, None, None
            else:
                raise ValueError("Model is not callable")

        results = {
            'moving': moving_tensor,
            'fixed': fixed_tensor,
            'warped': warped,
            'flow': flow,
            'flow_list': flow_list
        }

        # Apply flow to segmentations if available
        if 'seg' in moving_data or 'lab1' in moving_data:
            seg_key = 'seg' if 'seg' in moving_data else 'lab1'
            warped_seg = self.reg_models['nearest']([moving_data[seg_key], flow])
            results['moving_seg'] = moving_data[seg_key]
            results['warped_seg'] = warped_seg

            if 'seg' in fixed_data or 'lab2' in fixed_data:
                seg_key = 'seg' if 'seg' in fixed_data else 'lab2'
                results['fixed_seg'] = fixed_data[seg_key]

        return results


class BiomechanicalRegistrationWorkflow(RegistrationWorkflow):
    """Registration with biomechanical constraints"""

    def __init__(self, config):
        super().__init__(config)
        self.loss_nhe = None

    def setup(self):
        """Setup biomechanical models"""
        super().setup()

        # Setup Neo-Hookean elasticity loss
        self.loss_nhe = losses.local_grad_neo_hookean_dHCP(
            shape=tuple(self.config['imgsize']),
            return_maps=False,
            use_entropy=True,
            weight_entropy=0,
            normalize=False,
            return_everything=True
        )

    def run_registration(self, moving_data, fixed_data):
        """Run biomechanical registration"""

        # Prepare data
        x_in, _, moving_tensor, fixed_tensor = prepare_data_for_registration(
            moving_data, fixed_data)

        # Fine-tuning with biomechanical constraints
        if self.config.get('finetune_iterations', 0) > 0:
            self.model.train()
            optimizer = setup_optimizer(self.model, self.config.get('learning_rate', 3e-4))

            for iteration in range(self.config['finetune_iterations']):
                optimizer.zero_grad()

                output = self.model(x_in)
                warped, flow = output[0], output[1]

                # Combined loss: similarity + biomechanical constraint
                ncc_loss = losses.NCC().forward(warped, fixed_tensor)
                bio_loss = losses.local_grad_neo_hookean(
                    return_maps=False,
                    mode="original",
                    ratio=self.config.get('biomech_ratio', 5.0)
                ).forward(flow, flow)

                total_loss = ncc_loss + bio_loss
                total_loss.backward()
                optimizer.step()

                if self.config.get('verbose') and iteration % 10 == 0:
                    print(f"Iteration {iteration}: NCC={ncc_loss.item():.4f}, Bio={bio_loss.item():.4f}")

        # Final inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_in)
            warped, flow = output[0], output[1]

            # Get biomechanical parameters
            if len(output) > 2 and self.loss_nhe:
                biomech_params = self.loss_nhe.forward(-flow, output[-1])
                nhe, ratio, F, J = biomech_params
            else:
                nhe, ratio, F, J = None, None, None, None

        results = {
            'moving': moving_tensor,
            'fixed': fixed_tensor,
            'warped': warped,
            'flow': flow,
            'biomech_nhe': nhe,
            'biomech_ratio': ratio,
            'deformation_gradient': F,
            'jacobian': J
        }

        return results


class MultiAtlasSegmentationWorkflow(RegistrationWorkflow):
    """Multi-atlas segmentation workflow"""

    def run_multi_atlas_segmentation(self, atlas_dir, target_data):
        """Run multi-atlas segmentation"""

        atlas_files = sorted([f for f in os.listdir(atlas_dir) if f.endswith('.npz')])
        if not atlas_files:
            raise ValueError(f"No NPZ files found in {atlas_dir}")

        # Load target data
        if isinstance(target_data, str):
            target_data = load_npz_data(target_data)

        # Compute initial similarities to select best atlases
        if self.config.get('verbose'):
            print("Computing atlas similarities...")

        similarities = []
        for atlas_file in atlas_files[:min(len(atlas_files), self.config.get('n_atlases', 5) * 2)]:
            atlas_path = os.path.join(atlas_dir, atlas_file)
            atlas_data = load_npz_data(atlas_path)

            # Quick similarity check
            x_in, _, _, _ = prepare_data_for_registration(atlas_data, target_data)
            if x_in is not None:
                with torch.no_grad():
                    output = self.model(x_in)
                    similarity = losses.NCC().forward(output[0], target_data['vol']).item()
                    similarities.append((similarity, atlas_file, atlas_path))

        # Select top N atlases
        similarities.sort(key=lambda x: x[0], reverse=True)
        selected_atlases = similarities[:self.config.get('n_atlases', 5)]

        if self.config.get('verbose'):
            print(f"Selected {len(selected_atlases)} atlases:")
            for sim, name, _ in selected_atlases:
                print(f"  {name}: NCC = {sim:.4f}")

        # Register each atlas to target
        warped_segs = []
        final_similarities = []

        for similarity, atlas_file, atlas_path in selected_atlases:
            if self.config.get('verbose'):
                print(f"Registering {atlas_file}...")

            atlas_data = load_npz_data(atlas_path)

            # Run registration
            results = self.run_registration(atlas_data, target_data)

            if 'warped_seg' in results:
                warped_seg_cpu = utils.cpu(results['warped_seg'])
                warped_segs.append(warped_seg_cpu)

                # Calculate final similarity
                final_sim = losses.NCC_full().forward(
                    results['fixed'], results['warped']
                ).detach().cpu().numpy()
                final_similarities.append(final_sim)

        # Fuse segmentations
        if warped_segs:
            fusion_method = self.config.get('fusion_method', 'majority')
            fused_seg = multi_atlas_fusion(warped_segs, final_similarities, fusion_method)

            # Apply smoothing if requested
            smoothing = self.config.get('smoothing', 'none')
            if smoothing != 'none':
                fused_seg = smooth_segmentation(
                    fused_seg,
                    method=smoothing,
                    kernel_size=self.config.get('kernel_size', 3)
                )

            return {
                'target': target_data,
                'fused_segmentation': fused_seg,
                'individual_segs': warped_segs,
                'similarities': final_similarities,
                'atlas_files': [name for _, name, _ in selected_atlases]
            }

        return None

    def run_registration(self, moving_data, fixed_data):
        """Run single atlas registration"""
        return super().run_registration(moving_data, fixed_data)


def create_visualization(results, config, output_dir):
    """Create visualization plots"""

    if not results:
        return

    # Standard registration plots
    if 'warped' in results:
        plot_registration_results(
            results.get('moving'),
            results.get('fixed'),
            results['warped'],
            results.get('moving_seg'),
            results.get('fixed_seg'),
            results.get('warped_seg'),
            slice_idx=config.get('plot_slice'),
            save_path=os.path.join(output_dir, 'registration_results.png') if config.get('save_plots') else None
        )

    # Deformation grid
    if 'flow' in results and results['flow'] is not None:
        plot_deformation_grid_3d(results['flow'], config['imgsize'])
        if config.get('save_plots'):
            plt.savefig(os.path.join(output_dir, 'deformation_grid.png'), dpi=300)

    # Multi-atlas specific visualizations
    if 'fused_segmentation' in results:
        plt.figure(figsize=(15, 10))
        n_atlases = len(results.get('individual_segs', []))

        slice_idx = config.get('plot_slice', results['fused_segmentation'].shape[-1]//2)

        # Show individual atlas results
        for i, seg in enumerate(results.get('individual_segs', [])[:6]):  # Max 6 atlases
            plt.subplot(2, 4, i+1)
            plt.imshow(seg[..., slice_idx])
            plt.title(f"Atlas {i+1}")
            plt.axis('off')

        # Show fused result
        plt.subplot(2, 4, 7)
        plt.imshow(results['fused_segmentation'][..., slice_idx])
        plt.title("Fused Segmentation")
        plt.axis('off')

        plt.tight_layout()
        if config.get('save_plots'):
            plt.savefig(os.path.join(output_dir, 'multi_atlas_results.png'), dpi=300)
        plt.show()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Setup GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}")
    else:
        print("Using CPU")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration dictionary
    config = {
        'model_type': args.model_type,
        'imgsize': tuple(args.imgsize),
        'weights_path': args.weights,
        'nb_labels': args.nb_labels,
        'finetune_iterations': args.finetune_iterations,
        'learning_rate': args.learning_rate,
        'fusion_method': args.fusion_method,
        'n_atlases': args.n_atlases,
        'smoothing': args.smoothing,
        'kernel_size': args.kernel_size,
        'save_plots': args.save_plots,
        'plot_slice': args.plot_slice,
        'verbose': args.verbose,
        'model_params': {
            'batch_size': args.batch_size,
            'layer_mode': 'contracted' if args.model_type == 'casreg_nhe' else 'cascade'
        }
    }

    start_time = time.time()

    # Initialize workflow based on mode
    if args.mode == 'basic':
        workflow = BasicRegistrationWorkflow(config)
        workflow.setup()

        # Load data
        moving_data = load_npz_data(args.moving)
        fixed_data = load_npz_data(args.fixed)

        # Run registration
        results = workflow.run_registration(moving_data, fixed_data)

        # Evaluate
        metrics = workflow.evaluate_results(results)
        print(f"Registration completed. NCC: {metrics.get('ncc', 'N/A'):.4f}")
        if 'mean_dice' in metrics:
            print(f"Mean DICE: {metrics['mean_dice']:.4f}")

    elif args.mode == 'biomechanical':
        workflow = BiomechanicalRegistrationWorkflow(config)
        workflow.setup()

        # Load data
        moving_data = load_npz_data(args.moving)
        fixed_data = load_npz_data(args.fixed)

        # Run biomechanical registration
        results = workflow.run_registration(moving_data, fixed_data)

        # Evaluate
        metrics = workflow.evaluate_results(results)
        print(f"Biomechanical registration completed. NCC: {metrics.get('ncc', 'N/A'):.4f}")
        if 'folding_ratio' in metrics:
            print(f"Folding ratio: {metrics['folding_ratio']:.4f}")

    elif args.mode == 'multi_atlas':
        workflow = MultiAtlasSegmentationWorkflow(config)
        workflow.setup()

        # Run multi-atlas segmentation
        results = workflow.run_multi_atlas_segmentation(args.atlas_dir, args.target)

        if results:
            print(f"Multi-atlas segmentation completed using {len(results['atlas_files'])} atlases")
            print(f"Fusion method: {args.fusion_method}")

    elif args.mode == 'visualization':
        # Load existing results or run basic registration for visualization
        if args.moving and args.fixed:
            workflow = BasicRegistrationWorkflow(config)
            workflow.setup()

            moving_data = load_npz_data(args.moving)
            fixed_data = load_npz_data(args.fixed)
            results = workflow.run_registration(moving_data, fixed_data)
        else:
            print("Visualization mode requires --moving and --fixed arguments")
            return

    # Create visualizations
    if args.mode != 'multi_atlas' or (args.mode == 'multi_atlas' and results):
        create_visualization(results, config, args.output_dir)

    # Save results
    if results:
        # Prepare data for saving
        save_dict = {}

        if args.mode == 'multi_atlas':
            save_dict['segmentation'] = results.get('fused_segmentation')
            if 'target' in results and 'vol' in results['target']:
                save_dict['image'] = results['target']['vol']
        else:
            if 'warped' in results:
                save_dict['warped'] = results['warped']
            if 'warped_seg' in results:
                save_dict['segmentation'] = results['warped_seg']

        # Get metadata for NIFTI export
        metadata = {}
        if args.mode != 'multi_atlas' and args.fixed:
            fixed_npz = np.load(args.fixed)
            for key in ['spacing', 'origin', 'direction']:
                if key in fixed_npz:
                    metadata[key] = fixed_npz[key]
        elif args.mode == 'multi_atlas' and args.target:
            target_npz = np.load(args.target)
            for key in ['spacing', 'origin', 'direction']:
                if key in target_npz:
                    metadata[key] = target_npz[key]

        # Save as NIFTI
        subject_id = Path(args.fixed if args.fixed else args.target).stem
        save_results_nifti(save_dict, args.output_dir, subject_id, **metadata)

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()