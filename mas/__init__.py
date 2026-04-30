"""
Multi-Atlas Segmentation (MAS) Module

Unified registration and segmentation workflows for medical images.

Main components:
- mas_unified.py: Main script with all registration modes
- mas_examples.py: Usage examples and demonstrations  
- README.md: Comprehensive documentation

Usage:
    python mas_unified.py --mode <mode> [options]

Supported modes:
    - basic: Standard pairwise registration
    - biomechanical: Registration with Neo-Hookean constraints
    - multi_atlas: Multi-atlas label fusion
    - visualization: Plotting and analysis
"""

__version__ = "1.0.0"
__author__ = "Refactored from original mas/ scripts"
