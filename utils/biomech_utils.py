"""
Utility functions for biomechanical registration analysis and visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import matplotlib.colors as colors
try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("Warning: mayavi not available")
from scipy.ndimage import gaussian_filter1d
import os


def load_brain_structures(structures_file):
    """Load brain structure names from text file."""
    with open(structures_file, 'r') as f:
        return [line.strip() for line in f.readlines()]



def perc(data):
    """Apply percentile clipping to data."""
    data[data < np.percentile(data, 0)] = np.percentile(data, 0)
    data[data > np.percentile(data, 100)] = np.percentile(data, 100)
    return data


def norm_0_1(array):
    """Normalize array to 0-1 range."""
    return (array - array.min()) / (array.max() - array.min())


def make_3d_plot(data, seg, savename, crop, color=False, view="axial", vmin=0, vmax=1, colormap="jet"):
    """Create 3D plot using Mayavi."""
    if not MAYAVI_AVAILABLE:
        print(f"Warning: Cannot create 3D plot {savename} - mayavi not available")
        return

    try:
        array_data = np.copy(data)

        if crop:
            array_data = array_data[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
            seg = seg[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]

        mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))

        if color:
            src = mlab.pipeline.scalar_field(array_data)
            mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)
            mlab.colorbar(orientation="vertical")
        else:
            # Create contour surface
            contours = mlab.contour3d(array_data, contours=8, transparent=True, colormap=colormap)
            mlab.colorbar(orientation="vertical")

        mlab.savefig(savename)
        mlab.close()
    except Exception as e:
        print(f"Warning: Failed to create 3D plot {savename}: {e}")
        try:
            mlab.close()
        except:
            pass


def make_subplots(data, seg, vmin, vmax, cmap, savedir, savename):
    """Create subplot visualization."""
    axial = [25, 90, 110, 120, 140, 170]
    sagittal = [0, 40, 60, 70, 85, 98]
    coronal = [10, 35, 75, 90, 110, 160]

    fig, axes = plt.subplots(3, 6, figsize=(20, 12))

    # Axial slices
    for i, slice_idx in enumerate(axial):
        axes[0, i].imshow(data[slice_idx, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].contour(seg[slice_idx, :, :], colors='white', linewidths=0.5)
        axes[0, i].set_title(f'Axial {slice_idx}')
        axes[0, i].axis('off')

    # Sagittal slices
    for i, slice_idx in enumerate(sagittal):
        axes[1, i].imshow(data[:, slice_idx, :], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].contour(seg[:, slice_idx, :], colors='white', linewidths=0.5)
        axes[1, i].set_title(f'Sagittal {slice_idx}')
        axes[1, i].axis('off')

    # Coronal slices
    for i, slice_idx in enumerate(coronal):
        axes[2, i].imshow(data[:, :, slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, i].contour(seg[:, :, slice_idx], colors='white', linewidths=0.5)
        axes[2, i].set_title(f'Coronal {slice_idx}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(savedir, savename), dpi=300)
    plt.close()


def make_subplots_flow(img, flow, mask, vmin, vmax, cmap, savename):
    """Create flow field visualization."""
    axial = [70, 110, 130, 140, 155, 165]
    sagittal = [45, 55, 63, 66, 75, 98]
    coronal = [40, 70, 90, 110, 140, 150]

    fig, axes = plt.subplots(3, 6, figsize=(20, 12))

    # Create flow visualizations for each view
    for i, slice_idx in enumerate(axial):
        if i < len(axes[0]):
            # Flow field visualization with arrows
            s = 8  # stride for arrow display
            win = 3  # window for local mean subtraction

            V = flow[1, slice_idx, ::s, ::s]
            U = flow[2, slice_idx, ::s, ::s]

            # Local mean subtraction
            for x in range(V.shape[0]):
                for y in range(V.shape[1]):
                    V[x, y] = V[x, y] - V[max(0, x-win):min(V.shape[0], x+win),
                                         max(0, y-win):min(V.shape[1], y+win)].mean()
                    U[x, y] = U[x, y] - U[max(0, x-win):min(U.shape[0], x+win),
                                         max(0, y-win):min(U.shape[1], y+win)].mean()

            # Mask small values
            V[mask[slice_idx, ::s, ::s] == 0] = 1e-5 * V[mask[slice_idx, ::s, ::s] == 0]
            U[mask[slice_idx, ::s, ::s] == 0] = 1e-5 * U[mask[slice_idx, ::s, ::s] == 0]

            modulus = 1e-5 + 0.8**2 * (U**2 + V**2)**0.5
            modulus = (modulus - modulus.min()) / (modulus.max() - modulus.min())

            axes[0, i].imshow(img[slice_idx, :, :], cmap='gray', alpha=0.7)
            axes[0, i].quiver(np.arange(0, img.shape[2], s), np.arange(0, img.shape[1], s),
                             U, V, modulus, cmap=cmap, scale=50)
            axes[0, i].set_title(f'Axial {slice_idx}')
            axes[0, i].axis('off')

    plt.tight_layout()
    plt.savefig(savename, dpi=300)
    plt.close()


def hist_tissues(data, seg, namedata, vmin, vmax, savename=None):
    """Create histogram for tissue types."""
    def make_hist(data, seg, n, label, vmin, vmax):
        counts, bins = np.histogram(data[seg == n].flatten(), bins=1000, density=True, range=(vmin, vmax))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        smoothed = gaussian_filter1d(counts, sigma=2)
        plt.plot(bin_centers, smoothed, label=label, linewidth=2)

    plt.figure(figsize=(10, 6))
    make_hist(data, seg, 2, "CGM", vmin, vmax)  # Cortical Gray Matter
    make_hist(data, seg, 3, "WM", vmin, vmax)   # White Matter
    make_hist(data, seg, 4, "TH", vmin, vmax)   # Thalamus

    plt.xlabel(namedata)
    plt.ylabel('Density')
    plt.grid()
    plt.legend()

    if savename:
        plt.savefig(savename, dpi=300)
    plt.close()


def hist_structures(data, structure, structure_list, namedata, savename, cmap):
    """Create histogram for brain structures."""
    rows = []
    for i, structure_name in enumerate(structure_list, start=1):
        if structure_name not in ['CSF', 'Extra-cranial background', 'Intra-cranial background',
                                 'Lat. Ventricle R', 'Lat. Ventricle L']:
            st_data = data[structure == i]
            if len(st_data) > 0:
                pc05 = np.percentile(st_data, 5)
                pc25 = np.percentile(st_data, 25)
                pc50 = np.percentile(st_data, 50)
                pc75 = np.percentile(st_data, 75)
                pc95 = np.percentile(st_data, 95)
                rows.append([structure_name, pc05, pc25, pc50, pc75, pc95])

    if rows:
        # Create visualization
        import pandas as pd
        import seaborn as sns

        df = pd.DataFrame(rows, columns=['Structure', 'P5', 'P25', 'P50', 'P75', 'P95'])

        plt.figure(figsize=(15, 8))
        sns.heatmap(df[['P5', 'P25', 'P50', 'P75', 'P95']].T,
                   xticklabels=df['Structure'],
                   yticklabels=['5th', '25th', '50th', '75th', '95th'],
                   cmap=cmap, cbar=True)
        plt.title(f'Distribution of {namedata} across brain structures')
        plt.xlabel('Brain Structures')
        plt.ylabel('Percentiles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.close()


def ensure_figure_dir(base_dir, subdir=None):
    """Ensure figure directory exists and return path."""
    if subdir:
        fig_dir = os.path.join(base_dir, 'figures', subdir)
    else:
        fig_dir = os.path.join(base_dir, 'figures')

    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def save_results_to_file(results_dict, filename):
    """Save analysis results to text file."""
    with open(filename, 'w') as f:
        f.write("Biomechanical Registration Analysis Results\n")
        f.write("=" * 50 + "\n\n")

        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")