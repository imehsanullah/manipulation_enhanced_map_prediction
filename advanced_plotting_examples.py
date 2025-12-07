#!/usr/bin/env python3
"""
Advanced examples for using the MapDataVisualizer.
Shows various ways to customize and analyze the map data.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from plots import MapDataVisualizer


def example_1_analyze_height_distribution():
    """Analyze the distribution of heights across all cameras."""
    print("\n" + "="*70)
    print("Example 1: Height Distribution Analysis")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    hms_data = viz.load_hms_data(sample_dir)
    heights = hms_data['hms'][:, :, :, 0]  # (N, H, W)

    # Get non-zero heights
    non_zero_heights = heights[heights > 0]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero_heights.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Height (m)')
    plt.ylabel('Frequency')
    plt.title(f'Height Distribution - Sample {sample_dir.name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/height_distribution.png', dpi=150)
    print(f"Saved: plots/height_distribution.png")
    plt.close()

    # Print statistics
    print(f"\nHeight Statistics:")
    print(f"  Min: {non_zero_heights.min():.4f} m")
    print(f"  Max: {non_zero_heights.max():.4f} m")
    print(f"  Mean: {non_zero_heights.mean():.4f} m")
    print(f"  Std: {non_zero_heights.std():.4f} m")
    print(f"  Median: {np.median(non_zero_heights):.4f} m")


def example_2_semantic_class_distribution():
    """Analyze semantic class distribution."""
    print("\n" + "="*70)
    print("Example 2: Semantic Class Distribution")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    gt_data = viz.load_gt_data(sample_dir)
    semantic_2d = gt_data['semantic_2d']

    # Count pixels per class
    unique, counts = np.unique(semantic_2d, return_counts=True)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(unique, counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Semantic Class ID')
    plt.ylabel('Number of Pixels')
    plt.title(f'Semantic Class Distribution - Sample {sample_dir.name}')
    plt.xticks(unique)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/semantic_distribution.png', dpi=150)
    print(f"Saved: plots/semantic_distribution.png")
    plt.close()

    # Print class statistics
    print(f"\nClass Distribution:")
    for cls, count in zip(unique, counts):
        percentage = (count / semantic_2d.size) * 100
        print(f"  Class {cls:2d}: {count:6d} pixels ({percentage:5.2f}%)")


def example_3_compare_multiple_cameras():
    """Compare height maps from different camera viewpoints."""
    print("\n" + "="*70)
    print("Example 3: Multi-Camera Comparison")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    hms_data = viz.load_hms_data(sample_dir)

    # Select cameras to compare (different viewing angles)
    camera_indices = [0, 75, 150, 225]  # Different positions in the array

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, cam_idx in enumerate(camera_indices):
        hm = hms_data['hms'][cam_idx, :, :, 0]
        im = axes[i].imshow(hm, cmap='viridis', origin='upper')
        axes[i].set_title(f'Camera {cam_idx}')
        axes[i].set_xlabel('X (pixels)')
        axes[i].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[i], label='Height (m)')

    plt.suptitle(f'Multi-Camera Height Map Comparison - Sample {sample_dir.name}',
                fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/multi_camera_comparison.png', dpi=150)
    print(f"Saved: plots/multi_camera_comparison.png")
    plt.close()


def example_4_3d_voxel_analysis():
    """Analyze 3D voxel occupancy."""
    print("\n" + "="*70)
    print("Example 4: 3D Voxel Occupancy Analysis")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    gt_data = viz.load_gt_data(sample_dir)
    hm3d = gt_data['hm3d']  # (140, 200, 102)

    # Calculate occupancy per height level
    occupancy_per_level = (hm3d > 0.5).sum(axis=(0, 1))

    # Plot occupancy vs height
    height_levels = np.arange(len(occupancy_per_level)) * 0.005  # 5mm resolution
    plt.figure(figsize=(10, 6))
    plt.plot(height_levels, occupancy_per_level, linewidth=2)
    plt.fill_between(height_levels, occupancy_per_level, alpha=0.3)
    plt.xlabel('Height (m)')
    plt.ylabel('Number of Occupied Voxels')
    plt.title(f'Occupancy vs Height - Sample {sample_dir.name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/occupancy_vs_height.png', dpi=150)
    print(f"Saved: plots/occupancy_vs_height.png")
    plt.close()

    # Find peak occupancy height
    peak_idx = np.argmax(occupancy_per_level)
    peak_height = peak_idx * 0.005
    print(f"\nPeak occupancy at height: {peak_height:.4f} m")
    print(f"Number of occupied voxels at peak: {occupancy_per_level[peak_idx]}")


def example_5_depth_statistics():
    """Analyze depth image statistics across cameras."""
    print("\n" + "="*70)
    print("Example 5: Depth Image Analysis")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    hms_data = viz.load_hms_data(sample_dir)
    depths = hms_data['depths']  # (N, 480, 640)

    # Calculate mean depth per camera
    mean_depths = []
    for i in range(len(depths)):
        valid_depths = depths[i][depths[i] > 0]
        if len(valid_depths) > 0:
            mean_depths.append(valid_depths.mean())
        else:
            mean_depths.append(0)

    mean_depths = np.array(mean_depths)

    # Plot mean depth across cameras
    plt.figure(figsize=(12, 6))
    plt.plot(mean_depths, linewidth=2)
    plt.xlabel('Camera Index')
    plt.ylabel('Mean Depth (m)')
    plt.title(f'Mean Depth per Camera - Sample {sample_dir.name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/depth_per_camera.png', dpi=150)
    print(f"Saved: plots/depth_per_camera.png")
    plt.close()

    print(f"\nDepth Statistics Across All Cameras:")
    print(f"  Min mean depth: {mean_depths[mean_depths > 0].min():.4f} m")
    print(f"  Max mean depth: {mean_depths.max():.4f} m")
    print(f"  Overall mean: {mean_depths[mean_depths > 0].mean():.4f} m")


def example_6_cross_sections():
    """Visualize cross-sections of 3D voxel maps."""
    print("\n" + "="*70)
    print("Example 6: 3D Voxel Cross-Sections")
    print("="*70)

    viz = MapDataVisualizer()
    samples = viz.find_available_samples(0)
    sample_dir = samples[0]

    # Load data
    gt_data = viz.load_gt_data(sample_dir)
    hm3d = gt_data['hm3d']  # (140, 200, 102)

    # Create cross-sections
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # X-Z cross-section at middle Y
    middle_y = hm3d.shape[0] // 2
    ax = axes[0, 0]
    im = ax.imshow(hm3d[middle_y, :, :].T, cmap='viridis', origin='lower', aspect='auto')
    ax.set_title(f'X-Z Cross-Section (Y={middle_y})')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Z (height voxels)')
    plt.colorbar(im, ax=ax)

    # Y-Z cross-section at middle X
    middle_x = hm3d.shape[1] // 2
    ax = axes[0, 1]
    im = ax.imshow(hm3d[:, middle_x, :].T, cmap='viridis', origin='lower', aspect='auto')
    ax.set_title(f'Y-Z Cross-Section (X={middle_x})')
    ax.set_xlabel('Y (pixels)')
    ax.set_ylabel('Z (height voxels)')
    plt.colorbar(im, ax=ax)

    # X-Y cross-section at different Z levels
    z_levels = [10, 30]
    for i, z in enumerate(z_levels):
        ax = axes[1, i]
        im = ax.imshow(hm3d[:, :, z], cmap='viridis', origin='upper')
        ax.set_title(f'X-Y Cross-Section (Z={z}, {z*0.005:.3f}m)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'3D Voxel Cross-Sections - Sample {sample_dir.name}', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/cross_sections.png', dpi=150)
    print(f"Saved: plots/cross_sections.png")
    plt.close()


def main():
    """Run all examples."""
    import os
    os.makedirs('plots', exist_ok=True)

    print("\nRunning advanced plotting examples...")
    print("All plots will be saved to the 'plots/' directory.\n")

    example_1_analyze_height_distribution()
    example_2_semantic_class_distribution()
    example_3_compare_multiple_cameras()
    example_4_3d_voxel_analysis()
    example_5_depth_statistics()
    example_6_cross_sections()

    print("\n" + "="*70)
    print("All examples completed! Check the 'plots/' directory for outputs.")
    print("="*70)


if __name__ == '__main__':
    main()
