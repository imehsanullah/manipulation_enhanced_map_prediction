#!/usr/bin/env python3
"""
Visualize which pixels are visible vs occluded for each viewpoint.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def visualize_visible_vs_occluded(sample_dir: Path, save_path: str = None):
    """
    Create visualization showing visible vs occluded pixels.
    
    Args:
        sample_dir: Path to sample directory
        save_path: Path to save the figure
    """
    # Load ground truth data
    gt_file = sample_dir / "pre_action" / "gt_hms.npz"
    data = np.load(gt_file)
    instance_maps = data['instance_maps']
    
    n_views = len(instance_maps)
    map_height, map_width = instance_maps[0].shape
    total_pixels = map_height * map_width
    
    print(f"\nMap dimensions: {map_height} × {map_width} = {total_pixels:,} pixels")
    
    # Create figure with 2 columns per viewpoint
    fig = plt.figure(figsize=(20, 3 * n_views))
    
    for i in range(n_views):
        inst_map = instance_maps[i]
        visible_mask = (inst_map != -1)
        pixels_visible = np.sum(visible_mask)
        coverage_pct = (pixels_visible / total_pixels) * 100
        
        unique_instances = np.unique(inst_map[inst_map != -1])
        num_instances = len(unique_instances)
        
        # Left: Instance map
        ax1 = fig.add_subplot(n_views, 2, i*2 + 1)
        im1 = ax1.imshow(inst_map, cmap='tab20', interpolation='nearest', origin='upper')
        ax1.set_title(f'Viewpoint {i+1}: Instance Map\n'
                     f'{num_instances} instances | {coverage_pct:.1f}% coverage',
                     fontsize=11)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Instance ID', fraction=0.046, pad=0.04)
        
        # Right: Visible vs Occluded mask
        ax2 = fig.add_subplot(n_views, 2, i*2 + 2)
        
        # Create RGB image: green=visible, red=occluded
        rgb_mask = np.zeros((*visible_mask.shape, 3))
        rgb_mask[visible_mask] = [0, 0.7, 0]  # Green for visible
        rgb_mask[~visible_mask] = [0.7, 0, 0]  # Red for occluded
        
        ax2.imshow(rgb_mask)
        ax2.set_title(f'Viewpoint {i+1}: Visibility Mask\n'
                     f'✓ Visible: {pixels_visible:,} pixels ({coverage_pct:.1f}%)\n'
                     f'✗ Occluded: {total_pixels - pixels_visible:,} pixels ({100-coverage_pct:.1f}%)',
                     fontsize=11)
        ax2.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Visible (inst ≠ -1)'),
                          Patch(facecolor='red', label='Occluded (inst = -1)')]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Visible vs Occluded Pixels per Viewpoint\n'
                f'Green = pixels where objects are visible | Red = pixels that are occluded or empty',
                fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.close(fig)


def main():
    """Main function."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='Visualize visible vs occluded pixels')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to map_data directory')
    parser.add_argument('--job_id', type=int, default=0,
                       help='Job ID to analyze')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='Specific sample ID to analyze')
    parser.add_argument('--save', action='store_true',
                       help='Save visualization to disk')
    
    args = parser.parse_args()
    
    # Set data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = repo_root / "shelf_gym/data/map_data"
    
    # Find samples
    job_dir = data_dir / str(args.job_id)
    if not job_dir.exists():
        print(f"Job directory {job_dir} does not exist")
        return
    
    samples = sorted([d for d in job_dir.iterdir() if d.is_dir()])
    if not samples:
        print("No samples found!")
        return
    
    # Select sample
    if args.sample_id:
        sample_dir = data_dir / str(args.job_id) / args.sample_id
        if not sample_dir.exists():
            print(f"Sample {args.sample_id} not found!")
            return
    else:
        sample_dir = samples[0]
        print(f"Using sample: {sample_dir.name}")
    
    # Save path
    save_path = None
    if args.save:
        save_dir = script_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'visible_vs_occluded_{sample_dir.name}.png'
    
    # Run visualization
    visualize_visible_vs_occluded(sample_dir, save_path)


if __name__ == '__main__':
    main()
