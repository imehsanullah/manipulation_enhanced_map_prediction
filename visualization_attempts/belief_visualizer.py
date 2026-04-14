"""
Belief Visualizer Module
Can be imported and used programmatically in other scripts
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from matplotlib import pyplot as plt
import numpy as np
import os


class BeliefPlotter:
    """
    Standalone plotter for robot beliefs.
    Can be used independently without the full pipeline.
    """

    def __init__(self, n_classes=15, figsize=(12, 8), dpi=150):
        self.n_classes = n_classes
        self.figsize = figsize
        self.dpi = dpi

        # Import colormap
        from shelf_gym.utils.result_visualization_utils import get_my_cmap
        self.my_cmap = get_my_cmap(n_classes=n_classes)

    def plot_semantic_belief(self, semantic_map, mem_pipeline, title="Robot's Belief",
                            save_path=None, show=False):
        """
        Plot semantic belief map

        Args:
            semantic_map: Semantic map tensor (Dirichlet parameters)
            mem_pipeline: ManipulationEnhancedMapping instance (for helper functions)
            title: Plot title
            save_path: Path to save figure (optional)
            show: Whether to display the figure

        Returns:
            fig, ax: matplotlib figure and axes
        """
        sem_color, uncertainty_map, sem_labels, sem_conf = \
            mem_pipeline.get_semantic_rgb_image(semantic_map)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.imshow(sem_color)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def plot_belief_with_confidence(self, semantic_map, mem_pipeline, title="Robot's Belief",
                                   save_path=None, show=False):
        """
        Plot semantic belief with confidence map

        Args:
            semantic_map: Semantic map tensor
            mem_pipeline: ManipulationEnhancedMapping instance
            title: Plot title
            save_path: Path to save figure
            show: Whether to display

        Returns:
            fig, axes: matplotlib figure and axes
        """
        sem_color, uncertainty_map, sem_labels, sem_conf = \
            mem_pipeline.get_semantic_rgb_image(semantic_map)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Semantic belief
        axes[0].imshow(sem_color)
        axes[0].set_title(f'{title} - Semantic', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Confidence
        im = axes[1].imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'{title} - Confidence', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    def plot_complete_belief(self, occupancy_map, semantic_map, mem_pipeline,
                           title="Complete Belief", save_path=None, show=False):
        """
        Plot complete belief: semantic, confidence, and occupancy

        Args:
            occupancy_map: Occupancy map tensor
            semantic_map: Semantic map tensor
            mem_pipeline: ManipulationEnhancedMapping instance
            title: Plot title
            save_path: Path to save
            show: Whether to display

        Returns:
            fig, axes: matplotlib figure and axes
        """
        # Get semantic visualization
        sem_color, uncertainty_map, sem_labels, sem_conf = \
            mem_pipeline.get_semantic_rgb_image(semantic_map)

        # Get occupancy
        occ_prob = mem_pipeline.get_prob_map(occupancy_map).cpu().numpy()
        occ_2d = occ_prob[:, :, 10:].max(axis=-1)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Semantic belief
        axes[0, 0].imshow(sem_color)
        axes[0, 0].set_title('Semantic Belief', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Semantic confidence
        im1 = axes[0, 1].imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Semantic Confidence', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

        # Occupancy probability
        im2 = axes[1, 0].imshow(occ_2d, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Occupancy Probability', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

        # Uncertainty
        im3 = axes[1, 1].imshow(1 - sem_conf, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('Semantic Uncertainty', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes

    def plot_scene_and_belief(self, scene_image, semantic_map, mem_pipeline,
                            step=None, action_desc="", save_path=None, show=False):
        """
        Plot scene image and belief side-by-side (Figure 1 style)

        Args:
            scene_image: RGB image from camera
            semantic_map: Semantic map tensor
            mem_pipeline: ManipulationEnhancedMapping instance
            step: Step number (optional)
            action_desc: Action description (optional)
            save_path: Path to save
            show: Whether to display

        Returns:
            fig, axes: matplotlib figure and axes
        """
        sem_color, _, _, sem_conf = mem_pipeline.get_semantic_rgb_image(semantic_map)

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

        # Scene image
        ax_scene = fig.add_subplot(gs[0, :])
        if scene_image is not None:
            ax_scene.imshow(scene_image)
        title = 'Scene'
        if step is not None:
            title += f' - Step {step}'
        ax_scene.set_title(title, fontsize=14, fontweight='bold')
        ax_scene.axis('off')

        # Robot's belief
        ax_belief = fig.add_subplot(gs[1, :])
        ax_belief.imshow(sem_color)
        ax_belief.set_title("Robot's Belief", fontsize=14, fontweight='bold')
        ax_belief.axis('off')

        if action_desc:
            ax_belief.text(0.5, -0.05, f"Action: {action_desc}",
                         transform=ax_belief.transAxes, ha='center',
                         fontsize=11, style='italic',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Confidence
        ax_conf = fig.add_subplot(gs[2, 0])
        im1 = ax_conf.imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        ax_conf.set_title('Confidence', fontsize=12)
        ax_conf.axis('off')
        plt.colorbar(im1, ax=ax_conf, fraction=0.046)

        # Uncertainty
        ax_unc = fig.add_subplot(gs[2, 1])
        im2 = ax_unc.imshow(1 - sem_conf, cmap='viridis', vmin=0, vmax=1)
        ax_unc.set_title('Uncertainty', fontsize=12)
        ax_unc.axis('off')
        plt.colorbar(im2, ax=ax_unc, fraction=0.046)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax_scene

    def compare_beliefs(self, belief_maps, mem_pipeline, titles=None,
                       save_path=None, show=False):
        """
        Compare multiple beliefs side-by-side

        Args:
            belief_maps: List of semantic map tensors
            mem_pipeline: ManipulationEnhancedMapping instance
            titles: List of titles for each belief
            save_path: Path to save
            show: Whether to display

        Returns:
            fig, axes: matplotlib figure and axes
        """
        n_beliefs = len(belief_maps)

        if titles is None:
            titles = [f'Belief {i}' for i in range(n_beliefs)]

        fig, axes = plt.subplots(2, n_beliefs, figsize=(6*n_beliefs, 10))
        if n_beliefs == 1:
            axes = axes.reshape(-1, 1)

        for i, (belief_map, title) in enumerate(zip(belief_maps, titles)):
            sem_color, _, _, sem_conf = mem_pipeline.get_semantic_rgb_image(belief_map)

            # Semantic
            axes[0, i].imshow(sem_color)
            axes[0, i].set_title(title, fontsize=12, fontweight='bold')
            axes[0, i].axis('off')

            # Confidence
            im = axes[1, i].imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
            axes[1, i].set_title('Confidence', fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, axes


# Convenience functions for direct use
def quick_plot_belief(semantic_map, mem_pipeline, save_path='belief.png'):
    """Quick function to plot and save a belief"""
    plotter = BeliefPlotter()
    plotter.plot_semantic_belief(semantic_map, mem_pipeline,
                                save_path=save_path, show=False)


def quick_plot_belief_with_confidence(semantic_map, mem_pipeline, save_path='belief_conf.png'):
    """Quick function to plot belief with confidence"""
    plotter = BeliefPlotter()
    plotter.plot_belief_with_confidence(semantic_map, mem_pipeline,
                                       save_path=save_path, show=False)


def quick_plot_complete(occupancy_map, semantic_map, mem_pipeline, save_path='complete.png'):
    """Quick function to plot complete belief state"""
    plotter = BeliefPlotter()
    plotter.plot_complete_belief(occupancy_map, semantic_map, mem_pipeline,
                                save_path=save_path, show=False)


# Example usage
if __name__ == '__main__':
    print("Belief Visualizer Module")
    print("\nUsage example:")
    print("""
from belief_visualizer import BeliefPlotter, quick_plot_belief
from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping

# Initialize
mem = ManipulationEnhancedMapping(render=False, show_vis=False)
mem.reset_env()

# Get belief
cam_data, gt_data = mem.get_processed_array_and_gt_data()
previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(...)

# Quick plot
quick_plot_belief(previous_semantic_map, mem, 'my_belief.png')

# Or use plotter for more control
plotter = BeliefPlotter()
plotter.plot_complete_belief(previous_map, previous_semantic_map, mem,
                            title='My Custom Title',
                            save_path='custom.png')
""")
