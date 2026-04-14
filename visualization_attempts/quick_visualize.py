"""
Quick visualization script - simplified version
Just run: python quick_visualize.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
from shelf_gym.utils.model_evaluation_utils import get_igs_for_map
from matplotlib import pyplot as plt
import numpy as np
import torch
import os


def quick_visualize(n_steps=10):
    """Quick visualization with minimal setup"""

    # Get script directory for output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'quick_viz')
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing...")
    mem = ManipulationEnhancedMapping(render=False, show_vis=False)
    mem.reset_env()

    # Get initial data
    print("Getting initial observation...")
    cam_data, gt_data = mem.get_processed_array_and_gt_data()
    height_hms = np.array(cam_data['height_maps'])
    semantic_hms = np.array(cam_data['semantic_maps'])
    invalid_mask = height_hms[..., 0] == 0
    semantic_hms[invalid_mask] = mem.n_classes

    # Initialize belief
    print("Initializing belief...")
    previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
        torch.ones((1, 1, 204, 120, 200), device='cuda'))

    previous_views = []

    # Run and visualize
    print(f"\nRunning {n_steps} steps with visualization...")

    for step in range(n_steps):
        print(f"Step {step+1}/{n_steps}...", end=' ')

        # Visualize current belief
        sem_color, _, _, sem_conf = mem.get_semantic_rgb_image(previous_semantic_map)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Semantic belief
        axes[0].imshow(sem_color)
        axes[0].set_title(f"Robot's Belief - Step {step}", fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Confidence
        im = axes[1].imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Confidence - Step {step}', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'step_{step:02d}.png'),
                   dpi=120, bbox_inches='tight')
        plt.close()

        # Select next view
        igs, _ = get_igs_for_map(previous_map, mem.ig_calc, skip=1, use_alternative=True)
        igs[previous_views] = 0
        viewpoint = int(igs.argmax())

        # Execute observation
        previous_map, previous_semantic_map = mem.execute_observation(
            previous_views, viewpoint, previous_map, previous_semantic_map)

        print(f"viewpoint={viewpoint}, certainty={sem_conf.mean():.3f}")

    # Create summary figure
    print("\nCreating summary figure...")

    # Load selected frames
    frame_indices = [0, n_steps//3, 2*n_steps//3, n_steps-1]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, frame_idx in enumerate(frame_indices):
        # Load the saved image
        img_path = os.path.join(output_dir, f'step_{frame_idx:02d}.png')
        if os.path.exists(img_path):
            img = plt.imread(img_path)

            # Show full frame in top row
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Step {frame_idx}', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')

            # Show just semantic belief in bottom row
            # Re-load the data for this
            axes[1, i].text(0.5, 0.5, f't+{i}', ha='center', va='center',
                          fontsize=20, fontweight='bold')
            axes[1, i].axis('off')

    fig.suptitle('Belief Evolution Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    mem.close()

    print(f"\nDone! Visualizations saved to: {output_dir}/")
    print(f"  - Individual frames: step_00.png to step_{n_steps-1:02d}.png")
    print(f"  - Summary: summary.png")
    print(f"\nTo view results:")
    print(f"  Open: {os.path.join(output_dir, 'summary.png')}")

    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick belief visualization')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of steps to run (default: 10)')

    args = parser.parse_args()

    output_dir = quick_visualize(n_steps=args.steps)
