#!/usr/bin/env python3
"""
Debug version to identify exact tensor dimension issue
"""

import os
import sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models import losses
from models.models import CasReg_nhe
from utils.spatial import register_model
from data.data_utils import cpu

def debug_single_subject():
    """Debug processing of a single subject"""
    print("=== DEBUGGING BIOMECH PROCESSING ===")

    # Configuration
    imgsize = (192, 256, 256)

    # Load models
    print("Loading models...")
    weights_path1 = "../../registration_experiments/dHCP/forward/hybrid/CasReg2/results/weights/ncc0.947.pth.tar"
    weights_path2 = "/eos/jeodpp/data/projects/DIGLIFE/comteva/registration_experiments/dHCP/forward/nhe/CasReg2/results/weights/ncc0.924.pth.tar"

    model1 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5, mode="cascade", layer_mode="contracted")
    best_model1 = torch.load(weights_path1)['state_dict']
    model1.load_state_dict(best_model1)
    model1.cuda()

    model2 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5, mode="cascade", layer_mode="contracted")
    best_model2 = torch.load(weights_path2)['state_dict']
    model2.load_state_dict(best_model2)
    model2.cuda()

    # Setup registration tools
    reg_model = register_model(imgsize, 'nearest')
    reg_model_bilin = register_model(imgsize, 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()

    # Load template data
    template_path = "../../data/dHCP/npz/sub-CC00385XX15.npz"
    template_atlas_path = "../../data/dhcp-volumetric-atlas-groupwise/npz/dhcp_42.npz"

    template_data = np.load(template_path)
    template_atlas = np.load(template_atlas_path)

    template = template_data["vol2"][None, None, ...]
    template = np.ascontiguousarray(template)
    template = torch.tensor(template, requires_grad=True, dtype=torch.float32).cuda()

    # Setup loss function
    loss_nhe = losses.local_grad_neo_hookean_dHCP(
        shape=imgsize, return_maps=False, window=7,
        use_entropy=True, weight_entropy=0, normalize=False
    )

    # Load a single subject
    npz_file = "sub-CC00135AN12.npz"
    npz_dir = "../../data/dHCP/npz/"

    print(f"Loading subject: {npz_file}")
    data = np.load(os.path.join(npz_dir, npz_file))

    moving = data["vol1"][None, None, ...]
    moving_seg = data["lab1"][None, None, ...]
    fixed_seg_original = data["lab2"].copy()
    fixed_seg = data["lab2"].copy()

    print(f"Moving shape: {moving.shape}")
    print(f"Moving_seg shape: {moving_seg.shape}")
    print(f"Fixed_seg shape: {fixed_seg.shape}")
    print(f"Template shape: {template.shape}")

    # Process fixed_seg for loss computation
    fixed_seg[fixed_seg == 4] = 1
    fixed_seg[fixed_seg != 1] = 1e-5
    fixed_seg = gaussian_filter(fixed_seg, sigma=0.5)
    fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())
    fixed_seg = torch.tensor(fixed_seg)

    moving = torch.tensor(moving, requires_grad=True, dtype=torch.float32).cuda()
    moving_seg = torch.tensor(moving_seg, requires_grad=True, dtype=torch.float32).cuda()

    print(f"After conversion - Moving shape: {moving.shape}")
    print(f"After conversion - Template shape: {template.shape}")
    print(f"After conversion - Fixed_seg shape: {fixed_seg.shape}")

    # Registration 1: Biomechanical
    print("\n=== BIOMECHANICAL REGISTRATION ===")
    with torch.no_grad():
        x_in = torch.cat((moving, template), dim=1)
        print(f"Concatenated input shape: {x_in.shape}")

        output = model1(x_in)
        warped, flow = output[0], output[1]

        print(f"Flow shape: {flow.shape}")
        print(f"Output shapes: {[o.shape if hasattr(o, 'shape') else type(o) for o in output]}")

        # Examine the list outputs
        for i, o in enumerate(output):
            print(f"Output[{i}] type: {type(o)}")
            if isinstance(o, list):
                print(f"  List length: {len(o)}")
                for j, item in enumerate(o):
                    print(f"    Item[{j}] type: {type(item)} shape: {getattr(item, 'shape', 'no shape')}")
            elif hasattr(o, 'shape'):
                print(f"  Shape: {o.shape}")

        # Skip the problematic loss computation for now
        # Just create dummy values for testing the rest of the pipeline
        print("Skipping loss computation and creating dummy biomechanical parameters...")
        material_props = output[-1]  # Shape: [2, 191, 255, 256]
        print(f"Material properties tensor shape: {material_props.shape}")

        # Create dummy ratio for parameter extraction
        ratio = [material_props[0:1], material_props[1:2]]  # Split into 2 parts
        nhe = torch.tensor(0.0)  # Dummy NHE value
        F = [torch.zeros_like(flow[0, :1]), torch.zeros_like(flow[0, :1]), torch.zeros_like(flow[0, :1])]  # Dummy F
        J = torch.zeros_like(flow[0, 0])  # Dummy J

        try:

            print(f"NHE type/shape: {type(nhe)} {getattr(nhe, 'shape', 'no shape')}")
            print(f"Ratio type: {type(ratio)}")
            print(f"F type: {type(F)}")
            print(f"J type/shape: {type(J)} {getattr(J, 'shape', 'no shape')}")

            if hasattr(ratio, '__len__'):
                print(f"Ratio length: {len(ratio)}")
                for i, r in enumerate(ratio):
                    print(f"  Ratio[{i}] type/shape: {type(r)} {getattr(r, 'shape', 'no shape')}")
            else:
                print(f"Ratio shape: {getattr(ratio, 'shape', 'no shape')}")

            if hasattr(F, '__len__'):
                print(f"F length: {len(F)}")
                for i, f in enumerate(F):
                    print(f"  F[{i}] type/shape: {type(f)} {getattr(f, 'shape', 'no shape')}")
            else:
                print(f"F shape: {getattr(F, 'shape', 'no shape')}")

        except Exception as e:
            print(f"Error in loss computation: {e}")
            import traceback
            traceback.print_exc()
            return

    print("\n=== TRYING BIOMECHANICAL PARAMETER EXTRACTION ===")
    try:
        mu = np.zeros((192, 256, 256))
        kappa = np.zeros((192, 256, 256))

        print(f"Trying to access ratio[1]...")
        r1 = ratio[1]
        print(f"ratio[1] type/shape: {type(r1)} {getattr(r1, 'shape', 'no shape')}")

        print(f"Trying to access ratio[0]...")
        r0 = ratio[0]
        print(f"ratio[0] type/shape: {type(r0)} {getattr(r0, 'shape', 'no shape')}")

        print("Converting to cpu...")
        r1_cpu = cpu(r1)
        r0_cpu = cpu(r0)
        print(f"r1_cpu shape: {r1_cpu.shape}")
        print(f"r0_cpu shape: {r0_cpu.shape}")

        print("Trying assignment...")
        # Adjust for the actual tensor sizes - r1_cpu is already 3D (191, 255, 256)
        mu[:-1, :-1, :-1] = r1_cpu[:, :, :-1]  # Remove last dimension to match (191, 255, 255)
        kappa[:-1, :-1, :-1] = r0_cpu[:, :, :-1]  # Remove last dimension to match (191, 255, 255)

        print("SUCCESS: Biomechanical parameter extraction worked!")

    except Exception as e:
        print(f"Error in parameter extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_subject()