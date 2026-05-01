#!/usr/bin/env python3
"""
Script to analyze which viewpoint reveals the most instances in ground truth data.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_viewpoints(gt_data: dict) -> dict:
    """
    Analyze which viewpoint reveals the most instances.
    
    Args:
        gt_data: Dictionary containing instance_maps
        
    Returns:
        Dictionary with analysis results
    """
    instance_maps = gt_data['instance_maps']
    n_views = len(instance_maps)
    
    analysis = {
        'viewpoint_idx': [],
        'num_instances': [],
        'num_pixels_visible': [],
        'coverage_percentage': [],
        'unique_instance_ids': []
    }
    
    total_pixels = instance_maps[0].size
    
    for i in range(n_views):
        inst_map = instance_maps[i]
        
        # Count unique instances (excluding -1 which is background)
        unique_instances = np.unique(inst_map[inst_map != -1])
        num_instances = len(unique_instances)
        
        # Count total pixels with visible instances
        pixels_visible = np.sum(inst_map != -1)
        coverage_pct = (pixels_visible / total_pixels) * 100
        
        analysis['viewpoint_idx'].append(i)
        analysis['num_instances'].append(num_instances)
        analysis['num_pixels_visible'].append(pixels_visible)
        analysis['coverage_percentage'].append(coverage_pct)
        analysis['unique_instance_ids'].append(unique_instances)
    
    # Find best viewpoint
    best_idx = np.argmax(analysis['num_instances'])
    analysis['best_viewpoint'] = {
        'index': best_idx,
        'viewpoint_number': best_idx + 1,
        'num_instances': analysis['num_instances'][best_idx],
        'pixels_visible': analysis['num_pixels_visible'][best_idx],
        'coverage_percentage': analysis['coverage_percentage'][best_idx]
    }
    
    # Rank all viewpoints
    rankings = sorted(range(n_views), key=lambda i: analysis['num_instances'][i], reverse=True)
    analysis['rankings'] = rankings
    
    return analysis


def print_detailed_analysis(analysis: dict):
    """Print detailed analysis to console."""
    print("\n" + "="*80)
    print("VIEWPOINT ANALYSIS - Which View Reveals Most Objects?")
    print("="*80)
    
    best = analysis['best_viewpoint']
    print(f"\n🏆 BEST VIEWPOINT: #{best['viewpoint_number']}")
    print(f"   - Instances visible: {best['num_instances']}")
    print(f"   - Pixels visible: {best['pixels_visible']}")
    print(f"   - Coverage: {best['coverage_percentage']:.1f}%")
    
    print(f"\n📊 ALL VIEWPOINTS RANKED:")
    print("-"*80)
    print(f"{'Rank':<6} {'Viewpoint':<12} {'Instances':<12} {'Pixels':<12} {'Coverage':<12}")
    print("-"*80)
    
    for rank, idx in enumerate(analysis['rankings'], 1):
        print(f"{rank:<6} {idx+1:<12} {analysis['num_instances'][idx]:<12} "
              f"{analysis['num_pixels_visible'][idx]:<12} {analysis['coverage_percentage'][idx]:<12.1f}%")
    
    print("-"*80)
    
    # Show which instances are visible from each viewpoint
    print(f"\n🔍 INSTANCE VISIBILITY:")
    all_instances = set()
    for ids in analysis['unique_instance_ids']:
        all_instances.update(ids)
    
    print(f"Total unique instances in scene: {len(all_instances)}")
    
    # Find instances visible from all viewpoints vs only some
    instance_visibility = {}
    for inst_id in all_instances:
        visible_from = [i+1 for i, ids in enumerate(analysis['unique_instance_ids']) 
                       if inst_id in ids]
        instance_visibility[inst_id] = visible_from
    
    always_visible = [inst for inst, views in instance_visibility.items() 
                     if len(views) == len(analysis['rankings'])]
    sometimes_visible = [inst for inst, views in instance_visibility.items() 
                        if 0 < len(views) < len(analysis['rankings'])]
    rarely_visible = [inst for inst, views in instance_visibility.items() 
                     if len(views) == 1]
    
    print(f"\nAlways visible (all viewpoints): {len(always_visible)} instances")
    print(f"Sometimes visible (partial): {len(sometimes_visible)} instances")
    print(f"Rarely visible (only 1 viewpoint): {len(rarely_visible)} instances")
    
    if rarely_visible:
        print(f"\n⚠️  Hard-to-see instances (only visible from 1 viewpoint):")
        for inst in rarely_visible:
            views = instance_visibility[inst]
            print(f"   - Instance {inst}: only visible from Viewpoint {views[0]}")
    
    print("\n" + "="*80)


def main():
    """Main function with command-line interface."""
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='Analyze best viewpoint for instance visibility')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to map_data directory')
    parser.add_argument('--job_id', type=int, default=0,
                       help='Job ID to analyze')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='Specific sample ID to analyze')
    parser.add_argument('--save', action='store_true',
                       help='Save analysis plot to disk')
    
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
    
    # Load ground truth data
    gt_file = sample_dir / "pre_action" / "gt_hms.npz"
    if not gt_file.exists():
        print(f"File not found: {gt_file}")
        return
    
    print(f"Loading data from: {gt_file}")
    data = np.load(gt_file)
    gt_data = {
        'instance_maps': data['instance_maps']
    }
    
    # Analyze viewpoints
    analysis = analyze_viewpoints(gt_data)
    
    # Print detailed analysis
    print_detailed_analysis(analysis)
    
    # Save visualization if requested
    if args.save:
        save_dir = script_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f'viewpoint_analysis_{sample_dir.name}.png'
        
        # Create comparison plot
        n_views = len(analysis['viewpoint_idx'])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        colors = ['gold', 'silver', 'coral'] + ['lightblue'] * (n_views - 3)
        bars = ax1.bar(range(1, n_views + 1), analysis['num_instances'],
                      color=colors[:n_views], edgecolor='black')
        ax1.set_xlabel('Viewpoint Number')
        ax1.set_ylabel('Number of Instances Visible')
        ax1.set_title('Instances Visible per Viewpoint')
        ax1.set_xticks(range(1, n_views + 1))
        
        best = analysis['best_viewpoint']
        ax1.axvline(best['viewpoint_number'], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Best: VP {best["viewpoint_number"]}')
        ax1.legend()
        
        # Coverage plot
        ax2.bar(range(1, n_views + 1), analysis['coverage_percentage'],
               color='steelblue', edgecolor='black')
        ax2.set_xlabel('Viewpoint Number')
        ax2.set_ylabel('Coverage Percentage (%)')
        ax2.set_title('Map Coverage per Viewpoint')
        ax2.set_xticks(range(1, n_views + 1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")


if __name__ == '__main__':
    main()
