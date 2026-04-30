"""
Configuration file for biomechanical registration analysis.
"""

# Image dimensions
IMGSIZE = (192, 256, 256)

# Model configuration
MODEL_CONFIG = {
    'batch_size': 2,
    'coef_ratio': 1,
    'n_casc': 5,
    'mode': 'cascade',
    'layer_mode': 'contracted'
}

# Loss function parameters
LOSS_CONFIG = {
    'window': 7,
    'use_entropy': True,
    'weight_entropy': 0,
    'normalize': True,
    'return_maps': False
}

# Visualization parameters
VIZ_CONFIG = {
    'axial_slices': [25, 90, 110, 120, 140, 170],
    'sagittal_slices': [0, 40, 60, 70, 85, 98],
    'coronal_slices': [10, 35, 75, 90, 110, 160],
    'flow_axial_slices': [70, 110, 130, 140, 155, 165],
    'flow_sagittal_slices': [45, 55, 63, 66, 75, 98],
    'flow_coronal_slices': [40, 70, 90, 110, 140, 150],
    'dpi': 300,
    'figsize_large': (20, 12),
    'figsize_medium': (15, 8),
    'figsize_small': (10, 6)
}

# Processing parameters
PROCESSING_CONFIG = {
    'median_filter_size': 5,
    'flow_stride': 8,
    'flow_window': 3,
    'histogram_bins': 1000,
    'gaussian_sigma': 2
}

# File paths (relative to mas directory)
PATHS = {
    'brain_structures': 'brain_structures.txt',
    'figures_base': 'figures'
}