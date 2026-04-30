#!/usr/bin/env python3
"""
Quick test of the cleaned biomech script with one subject
"""

import os
import sys
import subprocess

def test_clean_script():
    """Test the cleaned script with one subject"""

    # Create a small test dataset with just one file
    test_npz_dir = "./test_npz_single"
    os.makedirs(test_npz_dir, exist_ok=True)

    # Copy one file for testing
    source_file = "../../data/dHCP/npz/sub-CC00135AN12.npz"
    test_file = os.path.join(test_npz_dir, "sub-CC00135AN12.npz")

    if not os.path.exists(test_file):
        subprocess.run(["cp", source_file, test_file], check=True)

    # Run the cleaned script
    cmd = [
        "python", "biomech_reg_to_template_clean.py",
        "--npz_files", test_npz_dir,
        "--in_weights", "../../registration_experiments/dHCP/forward/hybrid/CasReg2/results/weights/ncc0.947.pth.tar",
        "--in_weights2", "/eos/jeodpp/data/projects/DIGLIFE/comteva/registration_experiments/dHCP/forward/nhe/CasReg2/results/weights/ncc0.924.pth.tar",
        "--template", "../../data/dHCP/npz/sub-CC00385XX15.npz",
        "--template_atlas", "../../data/dhcp-volumetric-atlas-groupwise/npz/dhcp_42.npz",
        "--save_plot_dir", "./test_figures_single"
    ]

    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")

if __name__ == "__main__":
    test_clean_script()