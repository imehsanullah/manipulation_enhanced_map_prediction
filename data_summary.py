#!/usr/bin/env python3
"""
Utility script to summarize available map data.
"""

from plots import MapDataVisualizer
from pathlib import Path
import numpy as np


def summarize_data():
    """Print a summary of all available map data."""
    viz = MapDataVisualizer(data_dir="shelf_gym/data/map_data")

    print("=" * 70)
    print("MAP DATA SUMMARY")
    print("=" * 70)

    # Find all job directories
    data_dir = Path("shelf_gym/data/map_data")
    if not data_dir.exists():
        print(f"\nError: Directory {data_dir} does not exist!")
        return

    job_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"\nFound {len(job_dirs)} job directories")

    total_samples = 0

    for job_dir in job_dirs:
        job_id = job_dir.name
        samples = viz.find_available_samples(int(job_id))

        if samples:
            print(f"\nJob ID: {job_id}")
            print(f"  Number of samples: {len(samples)}")
            total_samples += len(samples)

            # Load first sample to get data shapes
            try:
                hms_data = viz.load_hms_data(samples[0])
                gt_data = viz.load_gt_data(samples[0])

                print(f"  First sample: {samples[0].name}")
                print(f"  Camera views per sample: {len(hms_data['hms'])}")
                print(f"  Height map resolution: {hms_data['hms'].shape[1]}x{hms_data['hms'].shape[2]}")
                print(f"  Depth image resolution: {hms_data['depths'].shape[1]}x{hms_data['depths'].shape[2]}")
                print(f"  3D voxel map shape: {gt_data['hm3d'].shape}")

                # Calculate some statistics
                heights = hms_data['hms'][:, :, :, 0]
                non_zero_heights = heights[heights > 0]
                if len(non_zero_heights) > 0:
                    print(f"  Height range: {non_zero_heights.min():.4f}m - {non_zero_heights.max():.4f}m")
                    print(f"  Mean height: {non_zero_heights.mean():.4f}m")

                # Semantic classes
                unique_classes = np.unique(gt_data['semantic_2d'])
                print(f"  Unique semantic classes: {len(unique_classes)}")
                print(f"  Class IDs: {sorted(unique_classes.tolist())}")

            except Exception as e:
                print(f"  Error loading sample: {e}")

    print("\n" + "=" * 70)
    print(f"TOTAL SAMPLES: {total_samples}")
    print("=" * 70)

    # List all samples in a table format
    print("\nDETAILED SAMPLE LIST:")
    print("-" * 70)
    print(f"{'Job ID':<10} {'Sample ID':<15} {'Path':<45}")
    print("-" * 70)

    for job_dir in job_dirs:
        job_id = job_dir.name
        samples = viz.find_available_samples(int(job_id))
        for sample in samples[:5]:  # Show first 5 samples per job
            print(f"{job_id:<10} {sample.name:<15} {str(sample):<45}")
        if len(samples) > 5:
            print(f"{'':10} ... and {len(samples) - 5} more samples")

    print("-" * 70)

    # Show available files in a sample
    if total_samples > 0:
        print("\nFILES IN SAMPLE DIRECTORY:")
        print("-" * 70)
        sample_dir = samples[0] / "pre_action"
        if sample_dir.exists():
            files = sorted(sample_dir.iterdir())
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name:<30} ({size_mb:,.2f} MB)")
        print("-" * 70)


if __name__ == '__main__':
    summarize_data()
