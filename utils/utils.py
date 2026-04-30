"""
Remaining utility functions that haven't been moved to specialized modules.
This file should be minimal after the reorganization.
"""

import numpy as np


# Only keep functions that are not moved to other modules
# Most functions have been moved to:
# - spatial.py: SpatialTransformer, register_model, pad_image
# - metrics.py: dice functions, dice_val, dice_val_substruct, etc.
# - math_ops.py: jacobian_determinant_vxm, MSE_torch, mk_grid
# - uncertainty.py: MC prediction functions, uncertainty calculations
# - training_utils.py: AverageMeter, adjust_learning_rate, save_checkpoint, Logger, etc.
# - visualization.py: comput_fig, mk_grid_img functions
# - image_utils.py: process_label, create_gaussian_flow
# - data/data_utils.py: multi_channel_labels, write2csv, cpu, norm_flow

# All major utility functions have been moved to appropriate modules.
# This file is kept for any remaining small utilities or for backward compatibility.