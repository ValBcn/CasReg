#!/usr/bin/env python3
"""
Example usage scripts for the unified MAS registration system
Demonstrates common use cases and workflows
"""

import os
import subprocess
import sys
from pathlib import Path


def run_basic_registration_example():
    """Example: Basic pairwise registration"""

    print("=== Basic Registration Example ===")

    # Example command
    cmd = [
        sys.executable, "mas_unified.py",
        "--mode", "basic",
        "--moving", "/path/to/moving_image.npz",
        "--fixed", "/path/to/fixed_image.npz",
        "--weights", "/path/to/casreg_weights.pth",
        "--output_dir", "./results/basic_registration",
        "--save_plots",
        "--verbose"
    ]

    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("- Load moving and fixed images from NPZ files")
    print("- Run basic registration using CasReg model")
    print("- Save warped images and plots to output directory")
    print("- Display registration metrics (NCC, DICE)")

    return cmd


def run_biomechanical_registration_example():
    """Example: Biomechanical registration with fine-tuning"""

    print("=== Biomechanical Registration Example ===")

    cmd = [
        sys.executable, "mas_unified.py",
        "--mode", "biomechanical",
        "--moving", "/path/to/fetal_brain_young.npz",
        "--fixed", "/path/to/fetal_brain_old.npz",
        "--weights", "/path/to/casreg_nhe_weights.pth",
        "--model_type", "casreg_nhe",
        "--output_dir", "./results/biomech_registration",
        "--finetune_iterations", "50",
        "--learning_rate", "1e-4",
        "--save_plots",
        "--verbose"
    ]

    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("- Use CasReg-NHE model with biomechanical constraints")
    print("- Fine-tune for 50 iterations with Neo-Hookean elasticity")
    print("- Output biomechanical parameters (strain, stress)")
    print("- Generate deformation grid visualizations")

    return cmd


def run_multi_atlas_segmentation_example():
    """Example: Multi-atlas brain segmentation"""

    print("=== Multi-Atlas Segmentation Example ===")

    cmd = [
        sys.executable, "mas_unified.py",
        "--mode", "multi_atlas",
        "--atlas_dir", "/path/to/dHCP_atlases",
        "--target", "/path/to/target_brain.npz",
        "--weights", "/path/to/casreg_weights.pth",
        "--output_dir", "./results/multi_atlas_seg",
        "--n_atlases", "10",
        "--fusion_method", "weighted",
        "--smoothing", "median",
        "--nb_labels", "8",
        "--save_plots",
        "--verbose"
    ]

    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("- Use 10 best matching atlases from dHCP dataset")
    print("- Register each atlas to target brain")
    print("- Fuse segmentations using weighted voting")
    print("- Apply median filtering for smoothing")
    print("- Save final segmentation as NIFTI")

    return cmd


def run_visualization_example():
    """Example: Create visualizations and plots"""

    print("=== Visualization Example ===")

    cmd = [
        sys.executable, "mas_unified.py",
        "--mode", "visualization",
        "--moving", "/path/to/moving.npz",
        "--fixed", "/path/to/fixed.npz",
        "--weights", "/path/to/model_weights.pth",
        "--output_dir", "./results/visualization",
        "--save_plots",
        "--plot_slice", "128"
    ]

    print("Command:", " ".join(cmd))
    print("\nThis will:")
    print("- Generate comprehensive visualization plots")
    print("- Show before/after registration comparison")
    print("- Create 3D deformation grid visualization")
    print("- Save all plots as high-resolution images")

    return cmd


def demonstrate_utility_functions():
    """Example: Using utility functions directly"""

    print("=== Using Utility Functions Directly ===")

    example_code = '''
import sys
sys.path.append('./utils')
from registration_utils import *

# Load data
moving_data = load_npz_data('/path/to/moving.npz')
fixed_data = load_npz_data('/path/to/fixed.npz')

# Initialize model
model = initialize_model(
    'casreg',
    imgsize=(192, 256, 256),
    weights_path='/path/to/weights.pth'
)

# Setup registration models
reg_models = setup_registration_models((192, 256, 256))

# Prepare data for registration
x_in, x_seg_in, moving_tensor, fixed_tensor = prepare_data_for_registration(
    moving_data, fixed_data
)

# Run registration
with torch.no_grad():
    warped, flow = model(x_in)[:2]

# Calculate metrics
metrics = calculate_metrics(warped, fixed_tensor, flow=flow)
print(f"NCC: {metrics['ncc']:.4f}")

# Create visualization
plot_registration_results(
    moving_tensor, fixed_tensor, warped,
    save_path='registration_result.png'
)
'''

    print("Python code example:")
    print(example_code)


def create_batch_processing_script():
    """Create example batch processing script"""

    batch_script = '''#!/bin/bash
# Example batch processing script for multiple subjects

ATLAS_DIR="/path/to/atlases"
WEIGHTS="/path/to/model_weights.pth"
OUTPUT_BASE="./results/batch_processing"

# Process all subjects in a directory
for subject_file in /path/to/subjects/*.npz; do
    subject_id=$(basename "$subject_file" .npz)
    output_dir="${OUTPUT_BASE}/${subject_id}"

    echo "Processing $subject_id..."

    python mas_unified.py \\
        --mode multi_atlas \\
        --atlas_dir "$ATLAS_DIR" \\
        --target "$subject_file" \\
        --weights "$WEIGHTS" \\
        --output_dir "$output_dir" \\
        --n_atlases 10 \\
        --fusion_method weighted \\
        --smoothing median \\
        --verbose

    echo "Completed $subject_id"
done

echo "Batch processing complete!"
'''

    with open('batch_process_example.sh', 'w') as f:
        f.write(batch_script)

    os.chmod('batch_process_example.sh', 0o755)
    print("Created batch_process_example.sh")


def main():
    """Show all examples"""

    print("="*60)
    print("UNIFIED MAS REGISTRATION SYSTEM - EXAMPLES")
    print("="*60)
    print()

    examples = [
        run_basic_registration_example,
        run_biomechanical_registration_example,
        run_multi_atlas_segmentation_example,
        run_visualization_example
    ]

    for example in examples:
        example()
        print()
        print("-" * 60)
        print()

    demonstrate_utility_functions()
    print()
    print("-" * 60)
    print()

    create_batch_processing_script()
    print()

    print("NOTES:")
    print("- Replace '/path/to/...' with actual file paths")
    print("- Adjust --imgsize based on your data dimensions")
    print("- Use --gpu -1 to force CPU usage")
    print("- Add --help to see all available options")
    print()
    print("For more help: python mas_unified.py --help")


if __name__ == "__main__":
    main()