#!/usr/bin/env python3
"""
Visualization script for map data stored in .npz files.
This script provides utilities to plot height maps, semantic maps, and ground truth data.
"""

import os
import numpy as np
import matplotlib
# Use non-interactive backend by default, can be overridden
if os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List
import argparse


class MapDataVisualizer:
    """Class to visualize map prediction data from .npz files."""

    def __init__(self, data_dir: str = "shelf_gym/data/map_data"):
        """
        Initialize the visualizer.

        Args:
            data_dir: Path to the map_data directory
        """
        self.data_dir = Path(data_dir)
        self.cmap = sns.color_palette("husl", 20)  # Color map for semantic visualization

    def find_available_samples(self, job_id: int = 0) -> List[Path]:
        """
        Find all available sample directories.

        Args:
            job_id: The job ID to search for samples

        Returns:
            List of paths to sample directories
        """
        job_dir = self.data_dir / str(job_id)
        if not job_dir.exists():
            print(f"Job directory {job_dir} does not exist")
            return []

        samples = sorted([d for d in job_dir.iterdir() if d.is_dir()])
        print(f"Found {len(samples)} samples in job {job_id}")
        return samples

    def load_hms_data(self, sample_dir: Path) -> dict:
        """
        Load height map data from hms.npz file.

        Args:
            sample_dir: Path to sample directory

        Returns:
            Dictionary containing height maps, semantic maps, depths, etc.
        """
        hms_file = sample_dir / "pre_action" / "hms.npz"
        if not hms_file.exists():
            raise FileNotFoundError(f"File not found: {hms_file}")

        data = np.load(hms_file)
        return {
            'hms': data['hms'],  # (N, 140, 200, 2) - height maps
            'dilated_hms': data['dilated_hms'],  # (N, 140, 200, 2) - dilated height maps
            'semantic_hms': data['semantic_hms'],  # (N, 140, 200) - semantic height maps
            'semantics': data['semantics'],  # (N, 480, 640, 2) - semantic images
            'depths': data['depths']  # (N, 480, 640) - depth images
        }

    def load_gt_data(self, sample_dir: Path) -> dict:
        """
        Load ground truth data from gt_hms.npz file.

        Args:
            sample_dir: Path to sample directory

        Returns:
            Dictionary containing ground truth maps
        """
        gt_file = sample_dir / "pre_action" / "gt_hms.npz"
        if not gt_file.exists():
            raise FileNotFoundError(f"File not found: {gt_file}")

        data = np.load(gt_file)
        return {
            'gt_hms': data['gt_hms'],  # (2, 140, 200) - ground truth occupancy and height
            'hm3d': data['hm3d'],  # (140, 200, 102) - 3D voxel height map
            'semantic_2d': data['semantic_2d'],  # (140, 200) - 2D semantic map
            'semantic_3d': data['semantic_3d'],  # (140, 200, 102) - 3D semantic voxel map
            'instance_maps': data['instance_maps']  # (12, 140, 200) - instance maps from 12 viewpoints
        }

    def plot_single_heightmap(self, hm: np.ndarray, camera_idx: int,
                              ax: Optional[plt.Axes] = None, title: str = None) -> plt.Axes:
        """
        Plot a single height map.

        Args:
            hm: Height map array (140, 200, 2) or (140, 200)
            camera_idx: Camera index for title
            ax: Matplotlib axes to plot on
            title: Custom title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Extract height channel if 3D
        if len(hm.shape) == 3:
            hm_display = hm[:, :, 0]
        else:
            hm_display = hm

        im = ax.imshow(hm_display, cmap='viridis', origin='upper')
        ax.set_title(title or f'Height Map - Camera {camera_idx}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Height (m)')

        return ax

    def plot_semantic_map(self, semantic_map: np.ndarray, ax: Optional[plt.Axes] = None,
                         title: str = "Semantic Map") -> plt.Axes:
        """
        Plot a semantic map with color coding.

        Args:
            semantic_map: Semantic map array (140, 200)
            ax: Matplotlib axes to plot on
            title: Title for the plot

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Create RGB image from semantic labels
        semantic_rgb = np.zeros((*semantic_map.shape, 3))
        unique_labels = np.unique(semantic_map)

        for label in unique_labels:
            if label >= 0 and label < len(self.cmap):
                mask = semantic_map == label
                semantic_rgb[mask] = self.cmap[int(label)]

        ax.imshow(semantic_rgb, origin='upper')
        ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        return ax

    def plot_depth_image(self, depth: np.ndarray, camera_idx: int,
                        ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot a depth image.

        Args:
            depth: Depth array (480, 640)
            camera_idx: Camera index for title
            ax: Matplotlib axes to plot on

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        im = ax.imshow(depth, cmap='gray', origin='upper')
        ax.set_title(f'Depth Image - Camera {camera_idx}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Depth')

        return ax

    def plot_camera_array_overview(self, hms_data: dict, camera_indices: List[int] = None,
                                   save_path: Optional[str] = None, show: bool = False):
        """
        Plot an overview of multiple camera views.

        Args:
            hms_data: Dictionary from load_hms_data()
            camera_indices: List of camera indices to plot (default: first 6)
            save_path: Path to save the figure
            show: Whether to display the plot (only works with interactive backend)
        """
        if camera_indices is None:
            camera_indices = list(range(min(6, len(hms_data['hms']))))

        n_cams = len(camera_indices)
        fig = plt.figure(figsize=(15, 3 * n_cams))
        gs = GridSpec(n_cams, 3, figure=fig, hspace=0.3, wspace=0.3)

        for i, cam_idx in enumerate(camera_indices):
            # Height map
            ax1 = fig.add_subplot(gs[i, 0])
            self.plot_single_heightmap(hms_data['hms'][cam_idx], cam_idx, ax=ax1,
                                      title=f'Height Map - Cam {cam_idx}')

            # Semantic map
            ax2 = fig.add_subplot(gs[i, 1])
            self.plot_semantic_map(hms_data['semantic_hms'][cam_idx], ax=ax2,
                                  title=f'Semantic Map - Cam {cam_idx}')

            # Depth image
            ax3 = fig.add_subplot(gs[i, 2])
            self.plot_depth_image(hms_data['depths'][cam_idx], cam_idx, ax=ax3)

        plt.suptitle('Camera Array Overview', fontsize=16, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_ground_truth_overview(self, gt_data: dict, save_path: Optional[str] = None, show: bool = False):
        """
        Plot ground truth data overview.

        Args:
            gt_data: Dictionary from load_gt_data()
            save_path: Path to save the figure
            show: Whether to display the plot (only works with interactive backend)
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Ground truth occupancy map
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt_data['gt_hms'][0], cmap='viridis', origin='upper')
        ax1.set_title('GT Occupancy Map')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Occupancy')

        # Ground truth height map
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(gt_data['gt_hms'][1], cmap='viridis', origin='upper')
        ax2.set_title('GT Height Map')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Height (m)')

        # 2D Semantic map
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_semantic_map(gt_data['semantic_2d'], ax=ax3, title='GT 2D Semantic Map')

        # 3D voxel map - show max projection along Z axis
        ax4 = fig.add_subplot(gs[1, 0])
        hm3d_max = gt_data['hm3d'].max(axis=2)
        im4 = ax4.imshow(hm3d_max, cmap='viridis', origin='upper')
        ax4.set_title('3D Voxel Map (Max Z Projection)')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='Max Occupancy')

        # 3D semantic map - show mode along Z axis
        ax5 = fig.add_subplot(gs[1, 1])
        from scipy import stats
        semantic_3d_mode = stats.mode(gt_data['semantic_3d'], axis=2, keepdims=False)[0]
        self.plot_semantic_map(semantic_3d_mode, ax=ax5, title='3D Semantic Map (Mode Z)')

        # 3D voxel height profile - show slice at middle Y
        ax6 = fig.add_subplot(gs[1, 2])
        middle_y = gt_data['hm3d'].shape[0] // 2
        im6 = ax6.imshow(gt_data['hm3d'][middle_y, :, :].T, cmap='viridis',
                        origin='lower', aspect='auto')
        ax6.set_title(f'3D Voxel Cross-Section (Y={middle_y})')
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Z (voxel height)')
        plt.colorbar(im6, ax=ax6, label='Occupancy')

        plt.suptitle('Ground Truth Data Overview', fontsize=16, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def analyze_viewpoints(self, gt_data: dict) -> dict:
        """
        Analyze which viewpoint reveals the most instances.

        Args:
            gt_data: Dictionary from load_gt_data() containing instance_maps

        Returns:
            Dictionary with analysis results
        """
        instance_maps = gt_data['instance_maps']
        n_views = len(instance_maps)
        
        analysis = {
            'viewpoint_idx': [],
            'num_instances': [],
            'num_pixels_visible': [],
            'coverage_percentage': []
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

    def plot_instance_maps(self, gt_data: dict, save_path: Optional[str] = None, show: bool = False):
        """
        Plot ground truth instance maps from multiple viewpoints with analysis.

        Args:
            gt_data: Dictionary from load_gt_data() containing instance_maps
            save_path: Path to save the figure
            show: Whether to display the plot (only works with interactive backend)
        """
        instance_maps = gt_data['instance_maps']
        
        # Analyze viewpoints
        analysis = self.analyze_viewpoints(gt_data)
        
        # instance_maps is a list/array of viewpoint maps
        n_views = len(instance_maps)
        n_cols = 3
        n_rows = int(np.ceil(n_views / n_cols)) + 1  # Extra row for summary
        
        fig = plt.figure(figsize=(20, 6 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
        
        # Use a categorical colormap for instance IDs
        cmap = plt.get_cmap('tab20')
        
        # Plot each viewpoint
        for i in range(n_views):
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            inst_map = instance_maps[i]
            
            # Display instance map
            im = ax.imshow(inst_map, cmap=cmap, interpolation='nearest', origin='upper')
            
            # Add ranking badge
            rank = analysis['rankings'].index(i) + 1
            badge_color = 'gold' if rank == 1 else 'silver' if rank == 2 else 'coral' if rank == 3 else 'lightgray'
            ax.text(0.98, 0.98, f'#{rank}', transform=ax.transAxes, fontsize=16,
                   fontweight='bold', ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, edgecolor='black'))
            
            ax.set_title(f'Viewpoint {i+1}\nInstances: {analysis["num_instances"][i]}\n'
                        f'Coverage: {analysis["coverage_percentage"][i]:.1f}%', 
                        fontsize=10)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Instance ID')
        
        # Summary plot in the remaining space
        ax_summary = fig.add_subplot(gs[-1, :])
        
        # Bar chart of instances per viewpoint
        colors = ['gold', 'silver', 'coral'] + ['lightblue'] * (n_views - 3)
        bars = ax_summary.bar(range(1, n_views + 1), analysis['num_instances'], 
                             color=colors[:n_views], edgecolor='black', linewidth=1.5)
        
        ax_summary.set_xlabel('Viewpoint Number', fontsize=12)
        ax_summary.set_ylabel('Number of Instances Visible', fontsize=12)
        ax_summary.set_title('Viewpoint Comparison: Which View Reveals Most Objects?', fontsize=14)
        ax_summary.set_xticks(range(1, n_views + 1))
        
        # Add value labels on bars
        for i, (bar, instances, coverage) in enumerate(zip(bars, analysis['num_instances'], 
                                                            analysis['coverage_percentage'])):
            height = bar.get_height()
            ax_summary.annotate(f'{instances}\n({coverage:.1f}%)',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        # Highlight best viewpoint
        best = analysis['best_viewpoint']
        ax_summary.axvline(best['viewpoint_number'], color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Best: VP {best["viewpoint_number"]}')
        ax_summary.legend()
        
        plt.suptitle(f'Ground Truth Instance Maps Analysis - {n_views} Viewpoints', 
                    fontsize=16, y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
            print(f"\nBest Viewpoint: #{best['viewpoint_number']}")
            print(f"  - Instances visible: {best['num_instances']}")
            print(f"  - Pixels visible: {best['pixels_visible']}")
            print(f"  - Coverage: {best['coverage_percentage']:.1f}%")
            print(f"\nRankings:")
            for rank, idx in enumerate(analysis['rankings'][:min(6, n_views)], 1):
                print(f"  {rank}. Viewpoint {idx+1}: {analysis['num_instances'][idx]} instances")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_comparison(self, sample_dir: Path, camera_idx: int = 0,
                       save_path: Optional[str] = None, show: bool = False):
        """
        Plot comparison between predicted and ground truth maps.

        Args:
            sample_dir: Path to sample directory
            camera_idx: Which camera view to show
            save_path: Path to save the figure
            show: Whether to display the plot (only works with interactive backend)
        """
        hms_data = self.load_hms_data(sample_dir)
        gt_data = self.load_gt_data(sample_dir)

        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Predicted height map
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_single_heightmap(hms_data['hms'][camera_idx], camera_idx, ax=ax1,
                                  title=f'Predicted Height - Cam {camera_idx}')

        # Ground truth height map
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(gt_data['gt_hms'][1], cmap='viridis', origin='upper')
        ax2.set_title('Ground Truth Height')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Height (m)')

        # Semantic comparison
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_semantic_map(gt_data['semantic_2d'], ax=ax3,
                              title='GT Semantic Map')

        plt.suptitle(f'Prediction vs Ground Truth - Sample {sample_dir.name}',
                    fontsize=16, y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_all_cameras_grid(self, hms_data: dict, max_cameras: int = 100,
                             save_path: Optional[str] = None, show: bool = False):
        """
        Plot all camera height maps in a grid.

        Args:
            hms_data: Dictionary from load_hms_data()
            max_cameras: Maximum number of cameras to plot
            save_path: Path to save the figure
            show: Whether to display the plot (only works with interactive backend)
        """
        n_cameras = min(len(hms_data['hms']), max_cameras)
        n_cols = 10
        n_rows = int(np.ceil(n_cameras / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
        axes = axes.flatten() if n_cameras > 1 else [axes]

        for i in range(n_cameras):
            ax = axes[i]
            hm = hms_data['hms'][i, :, :, 0]
            ax.imshow(hm, cmap='viridis', origin='upper')
            ax.set_title(f'Cam {i}', fontsize=8)
            ax.axis('off')

        # Hide unused subplots
        for i in range(n_cameras, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'All Camera Height Maps (Total: {n_cameras})', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def main():
    """Main function with command-line interface."""
    # Get script directory for relative paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='Visualize map data from .npz files')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to map_data directory (default: shelf_gym/data/map_data from repo root)')
    parser.add_argument('--job_id', type=int, default=0,
                       help='Job ID to visualize')
    parser.add_argument('--sample_id', type=str, default=None,
                       help='Specific sample ID to visualize (e.g., "000000008")')
    parser.add_argument('--mode', type=str, default='overview',
                       choices=['overview', 'ground_truth', 'comparison', 'all_cameras', 'instance_maps'],
                       help='Visualization mode')
    parser.add_argument('--camera_idx', type=int, default=0,
                       help='Camera index for comparison mode')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to disk')

    args = parser.parse_args()

    # Set data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = repo_root / "shelf_gym/data/map_data"

    # Initialize visualizer
    viz = MapDataVisualizer(str(data_dir))

    # Find samples
    samples = viz.find_available_samples(args.job_id)
    if not samples:
        print("No samples found!")
        return

    # Select sample
    if args.sample_id:
        sample_dir = viz.data_dir / str(args.job_id) / args.sample_id
        if not sample_dir.exists():
            print(f"Sample {args.sample_id} not found!")
            return
    else:
        sample_dir = samples[0]
        print(f"Using sample: {sample_dir.name}")

    # Generate plots based on mode
    save_path = None
    if args.save:
        save_dir = script_dir / 'plots'
        save_dir.mkdir(exist_ok=True)

    if args.mode == 'overview':
        hms_data = viz.load_hms_data(sample_dir)
        save_path = save_dir / f'overview_{sample_dir.name}.png' if args.save else None
        viz.plot_camera_array_overview(hms_data, save_path=str(save_path))

    elif args.mode == 'ground_truth':
        gt_data = viz.load_gt_data(sample_dir)
        save_path = save_dir / f'ground_truth_{sample_dir.name}.png' if args.save else None
        viz.plot_ground_truth_overview(gt_data, save_path=str(save_path))

    elif args.mode == 'comparison':
        save_path = save_dir / f'comparison_{sample_dir.name}.png' if args.save else None
        viz.plot_comparison(sample_dir, args.camera_idx, save_path=str(save_path))

    elif args.mode == 'all_cameras':
        hms_data = viz.load_hms_data(sample_dir)
        save_path = save_dir / f'all_cameras_{sample_dir.name}.png' if args.save else None
        viz.plot_all_cameras_grid(hms_data, save_path=str(save_path))

    elif args.mode == 'instance_maps':
        gt_data = viz.load_gt_data(sample_dir)
        save_path = save_dir / f'instance_maps_{sample_dir.name}.png' if args.save else None
        viz.plot_instance_maps(gt_data, save_path=str(save_path))


if __name__ == '__main__':
    main()
