"""
Script to recreate Figure 1 style visualizations from the paper.
Shows the progression of scene and robot's belief over time steps.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
from shelf_gym.utils.model_evaluation_utils import get_igs_for_map
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import os
from datetime import datetime


class BeliefVisualizer:
    def __init__(self, mem, output_dir=None):
        self.mem = mem
        # Get script directory for output paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if output_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(script_dir, f'belief_visualizations_{timestamp}')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Store snapshots for creating final figure
        self.snapshots = []

    def capture_snapshot(self, step, previous_map, previous_semantic_map,
                        viewpoint=None, action_type='observation', action_desc=""):
        """Capture current state for visualization"""

        # Get semantic visualization
        sem_color, uncertainty_map, sem_labels, sem_conf = self.mem.get_semantic_rgb_image(
            previous_semantic_map)

        # Get occupancy probability map
        occ_prob = self.mem.get_prob_map(previous_map).cpu().numpy()
        occ_2d = occ_prob[:, :, 10:].max(axis=-1)  # Max projection

        # Get camera view if viewpoint provided
        scene_image = None
        if viewpoint is not None:
            try:
                camera_data = self.mem.get_single_camera_array_heightmaps(viewpoint)
                if 'rgb_maps' in camera_data and len(camera_data['rgb_maps']) > 0:
                    scene_image = np.array(camera_data['rgb_maps'][0])
            except Exception as e:
                print(f"Could not capture scene image: {e}")

        # Store snapshot
        snapshot = {
            'step': step,
            'sem_color': sem_color,
            'sem_conf': sem_conf,
            'sem_labels': sem_labels,
            'uncertainty_map': uncertainty_map,
            'occ_2d': occ_2d,
            'scene_image': scene_image,
            'action_type': action_type,
            'action_desc': action_desc
        }

        self.snapshots.append(snapshot)
        return snapshot

    def save_individual_frame(self, snapshot, show_scene=True):
        """Save individual frame with scene and belief"""
        step = snapshot['step']

        if show_scene and snapshot['scene_image'] is not None:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

            # Scene image (top, spanning both columns)
            ax_scene = fig.add_subplot(gs[0, :])
            ax_scene.imshow(snapshot['scene_image'])
            ax_scene.set_title(f'Scene - Step {step}', fontsize=14, fontweight='bold')
            ax_scene.axis('off')

            # Robot's belief
            ax_belief = fig.add_subplot(gs[1, :])
        else:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
            ax_belief = fig.add_subplot(gs[0, :])

        # Semantic belief
        ax_belief.imshow(snapshot['sem_color'])
        ax_belief.set_title(f"Robot's Belief (Semantic) - Step {step}",
                           fontsize=14, fontweight='bold')
        ax_belief.axis('off')

        # Add action description
        if snapshot['action_desc']:
            ax_belief.text(0.5, -0.05, f"Action: {snapshot['action_desc']}",
                          transform=ax_belief.transAxes,
                          ha='center', fontsize=11, style='italic',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Confidence map
        row_idx = 2 if show_scene and snapshot['scene_image'] is not None else 1
        ax_conf = fig.add_subplot(gs[row_idx, 0])
        im_conf = ax_conf.imshow(snapshot['sem_conf'], cmap='hot', vmin=0, vmax=1)
        ax_conf.set_title(f'Confidence', fontsize=12)
        ax_conf.axis('off')
        plt.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)

        # Uncertainty map
        ax_unc = fig.add_subplot(gs[row_idx, 1])
        im_unc = ax_unc.imshow(1 - snapshot['sem_conf'], cmap='viridis', vmin=0, vmax=1)
        ax_unc.set_title(f'Uncertainty', fontsize=12)
        ax_unc.axis('off')
        plt.colorbar(im_unc, ax=ax_unc, fraction=0.046, pad=0.04)

        # Save
        filename = os.path.join(self.output_dir, f'frame_step_{step:03d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def create_figure1_style(self, snapshot_indices=None):
        """Create Figure 1 style comparison across multiple timesteps"""

        if snapshot_indices is None:
            # Select evenly spaced snapshots (like t, t+1, t+2 in paper)
            if len(self.snapshots) >= 3:
                snapshot_indices = [0, len(self.snapshots)//2, len(self.snapshots)-1]
            else:
                snapshot_indices = list(range(len(self.snapshots)))

        n_snapshots = len(snapshot_indices)

        # Create figure
        fig = plt.figure(figsize=(6*n_snapshots, 10))
        gs = fig.add_gridspec(3, n_snapshots, hspace=0.15, wspace=0.1,
                            height_ratios=[1, 0.8, 0.1])

        for col_idx, snap_idx in enumerate(snapshot_indices):
            if snap_idx >= len(self.snapshots):
                continue

            snapshot = self.snapshots[snap_idx]
            step = snapshot['step']

            # Scene image (top row)
            ax_scene = fig.add_subplot(gs[0, col_idx])
            if snapshot['scene_image'] is not None:
                ax_scene.imshow(snapshot['scene_image'])
                ax_scene.set_title('Scene', fontsize=14, fontweight='bold')
            else:
                ax_scene.text(0.5, 0.5, 'No scene\nimage',
                            ha='center', va='center', fontsize=12)
            ax_scene.axis('off')

            # Add annotation box for scene state
            scene_text = "Scene, no change" if snapshot['action_type'] == 'observation' else "Scene after manipulation"
            ax_scene.text(0.5, 1.05, scene_text,
                        transform=ax_scene.transAxes,
                        ha='center', fontsize=10, style='italic',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # Robot's belief (middle row)
            ax_belief = fig.add_subplot(gs[1, col_idx])
            ax_belief.imshow(snapshot['sem_color'])
            ax_belief.set_title("Robot's Belief", fontsize=14, fontweight='bold')
            ax_belief.axis('off')

            # Action description (bottom row)
            ax_action = fig.add_subplot(gs[2, col_idx])
            ax_action.axis('off')

            # Create action annotation
            action_text = snapshot['action_desc'] if snapshot['action_desc'] else f"Step {step}"

            if snapshot['action_type'] == 'push':
                bbox_color = 'lightcoral'
                action_label = 'Action: Move and Push'
            elif snapshot['action_type'] == 'observation':
                bbox_color = 'lightgreen'
                action_label = 'Action: View Change'
            else:
                bbox_color = 'wheat'
                action_label = f'Action: {snapshot["action_type"]}'

            ax_action.text(0.5, 0.5, action_label,
                          ha='center', va='center', fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor=bbox_color, alpha=0.8))

            # Add timestep label
            fig.text(0.5/n_snapshots + col_idx/n_snapshots, 0.02, f't+{col_idx}',
                    ha='center', fontsize=13, fontweight='bold')

        # Add main title
        fig.suptitle('Manipulation-Enhanced Mapping: Belief Evolution Over Time',
                    fontsize=16, fontweight='bold', y=0.98)

        # Add time step arrow
        if n_snapshots > 1:
            arrow = mpatches.FancyArrowPatch((0.15, 0.01), (0.85, 0.01),
                                            transform=fig.transFigure,
                                            arrowstyle='->', mutation_scale=30,
                                            linewidth=2, color='black')
            fig.patches.append(arrow)
            fig.text(0.5, 0.005, 'Time Step', ha='center', fontsize=12, fontweight='bold')

        # Save
        filename = os.path.join(self.output_dir, 'figure1_style_comparison.png')
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved Figure 1 style visualization: {filename}")

        return filename

    def create_video_frames(self):
        """Save all frames for creating a video"""
        print(f"Creating {len(self.snapshots)} video frames...")
        for snapshot in self.snapshots:
            self.save_individual_frame(snapshot, show_scene=True)
        print(f"All frames saved to {self.output_dir}")
        print("You can create a video using ffmpeg:")
        print(f"  ffmpeg -framerate 2 -pattern_type glob -i '{self.output_dir}/frame_*.png' -c:v libx264 -pix_fmt yuv420p belief_evolution.mp4")


def run_with_visualization(predefined_scene_dir=None, max_steps=20):
    """Run the pipeline with belief visualization"""

    # Output directory will be created by BeliefVisualizer
    output_dir = None

    print(f"Initializing Manipulation Enhanced Mapping...")
    mem = ManipulationEnhancedMapping(
        render=True,
        show_vis=False,
        use_uncertainty_informed_sampling=False
    )
    mem.reset_env()

    # Initialize visualizer (creates output_dir automatically)
    visualizer = BeliefVisualizer(mem)
    output_dir = visualizer.output_dir

    print("Starting run with visualization...")

    # Load predefined scene if provided
    if predefined_scene_dir:
        import pickle
        with open(predefined_scene_dir, 'rb') as f:
            arrangement = pickle.load(f)
        mem.restore_shelf_state(arrangement)
        print(f"Loaded scene from {predefined_scene_dir}")

    # Initial observation
    cam_data, gt_data = mem.get_processed_array_and_gt_data()
    start_positions, _ = mem.obj.update_obj_states(mem.current_obj_ids)

    # Convert and mask
    height_hms = np.array(cam_data['height_maps'])
    semantic_hms = np.array(cam_data['semantic_maps'])
    invalid_mask = height_hms[..., 0] == 0
    semantic_hms[invalid_mask] = mem.n_classes

    # Initialize belief
    previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
        torch.ones((1, 1, 204, 120, 200), device='cuda'))
    torch.cuda.empty_cache()

    # Capture initial belief
    visualizer.capture_snapshot(
        0, previous_map, previous_semantic_map,
        viewpoint=0,
        action_type='initial',
        action_desc="Initial belief (uniform prior)"
    )

    previous_views = []
    done_mapping = False

    # Main loop
    for step in range(1, max_steps + 1):
        print(f"\n=== Step {step}/{max_steps} ===")

        # Compute next viewpoint IG
        first_igs, _ = get_igs_for_map(previous_map, mem.ig_calc,
                                      skip=1, use_alternative=True)
        first_igs[previous_views] = 0
        max_obs_ig = float(first_igs.max())
        viewpoint = int(first_igs.argmax())
        print(f"Selected viewpoint={viewpoint}, IG={max_obs_ig:.3f}")

        # Check if we should consider pushing
        can_push = (step >= 3 and step < max_steps - 1 and not done_mapping)

        action_type = 'observation'
        action_desc = f"Observe from viewpoint {viewpoint}"

        if can_push:
            # Get second observation IG
            second_igs = mem.ig_calc.get_subsequent_igs_for_map(
                previous_map, [viewpoint], mem.ig_calc)
            second_igs[previous_views] = 0
            best_observation_ig = second_igs.max() + max_obs_ig

            # Get push candidates
            push_candidates = mem.get_possible_maps_push(
                previous_map, previous_semantic_map, num_points=mem.max_sampled_pushes)

            best_push_ig = 0.0
            if push_candidates['paths'] is not None:
                _, best_push, best_push_ig = mem.eval_push_igs(
                    push_candidates, previous_semantic_map,
                    use_delta_H=True, skip=5)

            print(f"IG comparison: obs={best_observation_ig:.3f} vs push={best_push_ig:.3f}")

            # Decide to push or observe
            if best_push_ig > best_observation_ig:
                print(">>> Performing PUSH action >>>")
                action_type = 'push'
                action_desc = f"Push action (IG={best_push_ig:.3f})"

                # Execute push
                from shelf_gym.utils.pushing_utils import execute_push
                execute_push(mem, push_candidates['paths'][best_push],
                           path_annotations=push_candidates['path_annotations'][best_push])

                # Update belief with predicted post-push map
                previous_views.clear()
                previous_map = push_candidates['possible_previous_maps'][best_push][None]
                previous_semantic_map = push_candidates['possible_semantic_maps'][best_push][None]

                # Capture post-push belief
                visualizer.capture_snapshot(
                    step, previous_map, previous_semantic_map,
                    viewpoint=None,
                    action_type='push',
                    action_desc=action_desc
                )
            else:
                # Perform observation
                print(f">>> Performing OBSERVATION action at viewpoint {viewpoint} >>>")
                previous_map, previous_semantic_map = mem.execute_observation(
                    previous_views, viewpoint, previous_map, previous_semantic_map)

                # Capture post-observation belief
                visualizer.capture_snapshot(
                    step, previous_map, previous_semantic_map,
                    viewpoint=viewpoint,
                    action_type='observation',
                    action_desc=action_desc
                )
        else:
            # Only observation allowed
            print(f">>> Performing OBSERVATION action at viewpoint {viewpoint} >>>")
            previous_map, previous_semantic_map = mem.execute_observation(
                previous_views, viewpoint, previous_map, previous_semantic_map)

            # Capture belief
            visualizer.capture_snapshot(
                step, previous_map, previous_semantic_map,
                viewpoint=viewpoint,
                action_type='observation',
                action_desc=action_desc
            )

        # Check for mapping completion
        sem_conf = mem.get_semantic_certainty(previous_semantic_map)
        certainly_mapped_fraction = mem.get_certainly_mapped_fraction(
            sem_conf, mem.prob_cutoff)
        done_mapping = certainly_mapped_fraction >= mem.stopping_criterion

        print(f"Mapped fraction: {certainly_mapped_fraction:.3f}")

        if done_mapping:
            print(">>> Mapping complete! >>>")
            break

    print("\n=== Generating visualizations ===")

    # Save individual frames
    print("Saving individual frames...")
    for snapshot in visualizer.snapshots[::2]:  # Save every other frame to reduce clutter
        visualizer.save_individual_frame(snapshot, show_scene=True)

    # Create Figure 1 style comparison
    print("\nCreating Figure 1 style comparison...")
    # Select interesting snapshots (start, middle with push if any, end)
    push_indices = [i for i, s in enumerate(visualizer.snapshots) if s['action_type'] == 'push']
    if push_indices:
        # If there were pushes, show: start, first push, end
        selected_indices = [0, push_indices[0], len(visualizer.snapshots) - 1]
    else:
        # Otherwise show: start, middle, end
        selected_indices = [0, len(visualizer.snapshots)//2, len(visualizer.snapshots) - 1]

    visualizer.create_figure1_style(selected_indices)

    # Create alternative view showing more steps
    if len(visualizer.snapshots) >= 5:
        print("\nCreating extended comparison (5 timesteps)...")
        extended_indices = np.linspace(0, len(visualizer.snapshots)-1, 5, dtype=int)

        fig = plt.figure(figsize=(25, 8))
        gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1)

        for col_idx, snap_idx in enumerate(extended_indices):
            snapshot = visualizer.snapshots[snap_idx]

            # Scene (if available)
            ax_scene = fig.add_subplot(gs[0, col_idx])
            if snapshot['scene_image'] is not None:
                ax_scene.imshow(snapshot['scene_image'])
            ax_scene.set_title(f'Step {snapshot["step"]}', fontsize=12, fontweight='bold')
            ax_scene.axis('off')

            # Belief
            ax_belief = fig.add_subplot(gs[1, col_idx])
            ax_belief.imshow(snapshot['sem_color'])
            ax_belief.axis('off')

            # Action type indicator
            color = 'red' if snapshot['action_type'] == 'push' else 'green'
            ax_belief.set_title(snapshot['action_type'].upper(),
                              fontsize=10, color=color, fontweight='bold')

        fig.suptitle('Belief Evolution: Extended View', fontsize=16, fontweight='bold')
        extended_filename = os.path.join(output_dir, 'extended_comparison.png')
        plt.savefig(extended_filename, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {extended_filename}")

    mem.close()

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total snapshots captured: {len(visualizer.snapshots)}")
    print(f"{'='*60}")

    return visualizer, output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize robot belief during MEM')
    parser.add_argument('--scene', type=str, default=None,
                       help='Path to predefined scene file (.p)')
    parser.add_argument('--steps', type=int, default=20,
                       help='Maximum number of steps to run')
    parser.add_argument('--scene-dir', type=str,
                       default='./data/Hard_scenes/scenes/',
                       help='Directory containing scene files')
    parser.add_argument('--scene-id', type=int, default=None,
                       help='Scene ID to load from scene-dir')

    args = parser.parse_args()

    # Determine scene path
    scene_path = None
    if args.scene:
        scene_path = args.scene
    elif args.scene_id is not None:
        scene_path = os.path.join(args.scene_dir, f'scene_data_{args.scene_id}.p')
        if not os.path.exists(scene_path):
            print(f"Warning: Scene file not found: {scene_path}")
            scene_path = None

    if scene_path:
        print(f"Using predefined scene: {scene_path}")
    else:
        print("Using randomly generated scene")

    # Run with visualization
    visualizer, output_dir = run_with_visualization(
        predefined_scene_dir=scene_path,
        max_steps=args.steps
    )

    print(f"\nTo create a video from the frames, run:")
    print(f"ffmpeg -framerate 2 -pattern_type glob -i '{output_dir}/frame_*.png' \\")
    print(f"       -c:v libx264 -pix_fmt yuv420p {output_dir}/belief_evolution.mp4")
