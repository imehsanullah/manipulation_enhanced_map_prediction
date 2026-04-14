#!/usr/bin/env python3
"""
Example script showing how to use the MapDataVisualizer class.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plots import MapDataVisualizer

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize the visualizer - path is relative to repo root
    repo_root = os.path.dirname(script_dir)
    data_dir = os.path.join(repo_root, "shelf_gym/data/map_data")
    viz = MapDataVisualizer(data_dir=data_dir)

    # Find available samples
    samples = viz.find_available_samples(job_id=0)

    if not samples:
        print("No samples found! Please ensure data exists in shelf_gym/data/map_data/0/")
        return

    # Use the first sample
    sample_dir = samples[0]
    print(f"\nVisualizing sample: {sample_dir}")

    # Example 1: Load and plot camera array overview
    print("\n1. Loading height map data...")
    hms_data = viz.load_hms_data(sample_dir)
    print(f"   - Found {len(hms_data['hms'])} camera views")
    print(f"   - Height map shape: {hms_data['hms'].shape}")
    print(f"   - Semantic map shape: {hms_data['semantic_hms'].shape}")

    # print("\n2. Plotting camera array overview (first 6 cameras)...")
    # viz.plot_camera_array_overview(hms_data, camera_indices=[0, 50, 100, 150, 200, 250],
    #                                save_path='plots/camera_array_overview.png')

    # Example 2: Load and plot ground truth data
    print("\n3. Loading ground truth data...")
    gt_data = viz.load_gt_data(sample_dir)
    print(f"   - GT height map shape: {gt_data['gt_hms'].shape}")
    print(f"   - 3D voxel map shape: {gt_data['hm3d'].shape}")

    print("\n4. Plotting ground truth overview...")
    viz.plot_ground_truth_overview(gt_data, save_path=os.path.join(plots_dir, 'ground_truth_overview.png'))

    # Example 3: Plot comparison
    print("\n5. Plotting comparison between prediction and ground truth...")
    viz.plot_comparison(sample_dir, camera_idx=0, save_path=os.path.join(plots_dir, 'comparison.png'))

    # # Example 4: Plot all cameras in a grid
    # print("\n6. Plotting all camera views in a grid...")
    # viz.plot_all_cameras_grid(hms_data, max_cameras=50, save_path=os.path.join(plots_dir, 'all_cameras_grid.png'))

    print(f"\nDone! All visualizations saved to '{plots_dir}/' directory.")

if __name__ == '__main__':
    main()
