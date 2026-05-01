#!/usr/bin/env python3
"""
Script to explain how map coverage is calculated per viewpoint.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def explain_coverage_calculation(sample_dir: Path, save_path: str = None):
    """
    Create a detailed visualization explaining coverage calculation.
    
    Args:
        sample_dir: Path to sample directory
        save_path: Path to save the explanation figure
    """
    # Load ground truth data
    gt_file = sample_dir / "pre_action" / "gt_hms.npz"
    data = np.load(gt_file)
    instance_maps = data['instance_maps']
    
    n_views = len(instance_maps)
    map_height, map_width = instance_maps[0].shape
    total_pixels = map_height * map_width
    
    print("\n" + "="*80)
    print("HOW MAP COVERAGE IS CALCULATED")
    print("="*80)
    print(f"\nMap dimensions: {map_height} × {map_width} = {total_pixels:,} total pixels")
    print(f"Number of viewpoints: {n_views}")
    
    # Create detailed figure
    fig = plt.figure(figsize=(20, 12))
    
    # For each viewpoint, show the calculation
    for i in range(n_views):
        inst_map = instance_maps[i]
        
        # Create binary mask: which pixels have visible instances?
        visible_mask = (inst_map != -1)
        pixels_visible = np.sum(visible_mask)
        coverage_pct = (pixels_visible / total_pixels) * 100
        
        # Count instances
        unique_instances = np.unique(inst_map[inst_map != -1])
        num_instances = len(unique_instances)
        
        # Create subplot
        ax = fig.add_subplot(2, 3, i+1)
        
        # Show the instance map
        im = ax.imshow(inst_map, cmap='tab20', interpolation='nearest', origin='upper')
        ax.set_title(f'Viewpoint {i+1}\n'
                    f'Coverage: {coverage_pct:.1f}% ({pixels_visible:,}/{total_pixels:,} pixels)\n'
                    f'Instances: {num_instances}',
                    fontsize=10)
        ax.axis('off')
        
        # Add annotation explaining the calculation
        explanation = (
            f"Step 1: Count pixels where inst_map ≠ -1\n"
            f"  → Visible pixels: {pixels_visible:,}\n"
            f"Step 2: Divide by total pixels\n"
            f"  → {pixels_visible:,} / {total_pixels:,} = {coverage_pct/100:.3f}\n"
            f"Step 3: Convert to percentage\n"
            f"  → {coverage_pct/100:.3f} × 100 = {coverage_pct:.1f}%"
        )
        
        # Add text box with calculation
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=7,
               verticalalignment='top', fontfamily='monospace',
               bbox=props)
    
    plt.suptitle('Map Coverage Calculation Explained\n'
                f'Each viewpoint renders a {map_height}×{map_width} top-down map. '
                f'Coverage = (pixels with visible objects) / (total pixels) × 100%',
                fontsize=14, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved explanation to: {save_path}")
    
    plt.close(fig)
    
    # Print detailed calculation for first viewpoint
    print(f"\n{'='*80}")
    print(f"DETAILED EXAMPLE: Viewpoint 1")
    print(f"{'='*80}")
    
    inst_map = instance_maps[0]
    visible_mask = (inst_map != -1)
    pixels_visible = np.sum(visible_mask)
    coverage_pct = (pixels_visible / total_pixels) * 100
    
    print(f"\n1. Instance map shape: {inst_map.shape}")
    print(f"2. Total pixels in map: {map_height} × {map_width} = {total_pixels:,}")
    print(f"3. Pixels with instances (inst_map ≠ -1): {pixels_visible:,}")
    print(f"4. Pixels without instances (inst_map = -1): {total_pixels - pixels_visible:,}")
    print(f"\n5. Coverage calculation:")
    print(f"   coverage = (visible pixels) / (total pixels) × 100")
    print(f"   coverage = ({pixels_visible:,}) / ({total_pixels:,}) × 100")
    print(f"   coverage = {pixels_visible/total_pixels:.4f} × 100")
    print(f"   coverage = {coverage_pct:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"WHAT DOES THIS MEAN?")
    print(f"{'='*80}")
    print(f"""
Coverage represents how much of the shelf surface is VISIBLE from that viewpoint.

- High coverage (e.g., 36%) = Camera sees most of the shelf surface
- Low coverage (e.g., 29%) = Camera sees less of the shelf surface

Why the difference?
1. Occlusion: Objects block the view of other objects/surface
2. Distance: Farther cameras may see more (or less) depending on angle
3. Perspective: Different angles reveal different hidden areas

Note: Coverage ≠ Number of instances!
- A viewpoint can see many instances but low coverage (scattered objects)
- A viewpoint can see few instances but high coverage (large visible surface)
""")


def main():
    """Main function."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='Explain how map coverage is calculated')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to map_data directory')
    parser.add_argument('--job_id', type=int, default=0,
                       help='Job ID to analyze')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='Specific sample ID to analyze')
    parser.add_argument('--save', action='store_true',
                       help='Save explanation to disk')
    
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
        save_path = save_dir / f'coverage_explanation_{sample_dir.name}.png'
    
    # Run explanation
    explain_coverage_calculation(sample_dir, save_path)


if __name__ == '__main__':
    main()
