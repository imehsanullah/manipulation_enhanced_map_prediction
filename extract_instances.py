"""
Script to extract instance segmentation from semantic belief maps
Uses connected components analysis to separate instances of the same class
"""

import matplotlib
matplotlib.use('Agg')

from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from skimage import measure
import os


class InstanceExtractor:
    """Extract instance segmentation from semantic beliefs"""

    def __init__(self, mem_pipeline):
        self.mem = mem_pipeline
        self.n_classes = mem_pipeline.n_classes

    def extract_instances(self, semantic_map, confidence_threshold=0.2, min_pixels=10):
        """
        Extract instance segmentation from semantic belief

        Args:
            semantic_map: Semantic map tensor (Dirichlet parameters)
            confidence_threshold: Minimum confidence to consider a pixel
            min_pixels: Minimum pixels for a valid instance

        Returns:
            instance_map: 2D array with unique instance IDs
            instance_info: List of dicts with instance metadata
        """
        # Get semantic labels and confidence
        sem_color, _, sem_labels, sem_conf = self.mem.get_semantic_rgb_image(semantic_map)

        # Filter by confidence
        sem_labels_filtered = np.where(sem_conf >= confidence_threshold, sem_labels, -1)

        # Initialize instance map (0 = background)
        instance_map = np.zeros_like(sem_labels_filtered, dtype=np.int32)
        instance_info = []

        next_instance_id = 1

        # Process each semantic class
        for class_id in range(self.n_classes):
            # Get mask for this class
            class_mask = (sem_labels_filtered == class_id)

            if not class_mask.any():
                continue

            # Find connected components (separate instances)
            labeled_components, num_components = ndimage.label(class_mask)

            # Process each connected component as a separate instance
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_components == component_id)
                num_pixels = component_mask.sum()

                # Filter small components (noise)
                if num_pixels < min_pixels:
                    continue

                # Assign unique instance ID
                instance_map[component_mask] = next_instance_id

                # Get bounding box
                rows, cols = np.where(component_mask)
                bbox = [cols.min(), rows.min(), cols.max(), rows.max()]

                # Get average confidence for this instance
                instance_conf = sem_conf[component_mask].mean()

                # Store instance info
                instance_info.append({
                    'instance_id': next_instance_id,
                    'class_id': int(class_id),
                    'num_pixels': int(num_pixels),
                    'bbox': bbox,
                    'confidence': float(instance_conf),
                    'centroid': [float(cols.mean()), float(rows.mean())]
                })

                next_instance_id += 1

        return instance_map, instance_info

    def visualize_instances(self, instance_map, instance_info, semantic_map,
                          save_path='instances.png'):
        """
        Visualize instance segmentation

        Args:
            instance_map: 2D array with instance IDs
            instance_info: List of instance metadata dicts
            semantic_map: Original semantic map for comparison
            save_path: Where to save the visualization
        """
        # Get semantic visualization
        sem_color, _, _, sem_conf = self.mem.get_semantic_rgb_image(semantic_map)

        # Create instance visualization with random colors
        np.random.seed(42)
        instance_colors = np.random.randint(0, 255, size=(instance_map.max() + 1, 3))
        instance_colors[0] = [0, 0, 0]  # Background is black

        instance_vis = instance_colors[instance_map]

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Semantic segmentation
        axes[0, 0].imshow(sem_color)
        axes[0, 0].set_title('Semantic Segmentation (Class-based)',
                            fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Instance segmentation
        axes[0, 1].imshow(instance_vis)
        axes[0, 1].set_title(f'Instance Segmentation ({len(instance_info)} instances)',
                            fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Draw bounding boxes
        for info in instance_info:
            bbox = info['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0], bbox[3] - bbox[1],
                                fill=False, edgecolor='red', linewidth=1)
            axes[0, 1].add_patch(rect)
            # Add instance ID label
            axes[0, 1].text(bbox[0], bbox[1] - 2,
                          f"#{info['instance_id']}",
                          color='yellow', fontsize=8, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

        # Instance IDs map
        axes[1, 0].imshow(instance_map, cmap='nipy_spectral', interpolation='nearest')
        axes[1, 0].set_title('Instance ID Map', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Confidence map
        im = axes[1, 1].imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('Confidence Map', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")

    def print_instance_summary(self, instance_info):
        """Print summary of detected instances"""
        print(f"\n{'='*60}")
        print(f"INSTANCE SEGMENTATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total instances detected: {len(instance_info)}")
        print(f"\n{'ID':<5} {'Class':<8} {'Pixels':<8} {'Confidence':<12} {'Centroid':<15}")
        print(f"{'-'*60}")

        for info in instance_info:
            print(f"{info['instance_id']:<5} "
                  f"{info['class_id']:<8} "
                  f"{info['num_pixels']:<8} "
                  f"{info['confidence']:<12.3f} "
                  f"({info['centroid'][0]:.1f}, {info['centroid'][1]:.1f})")

        # Count instances per class
        from collections import Counter
        class_counts = Counter([info['class_id'] for info in instance_info])

        print(f"\n{'='*60}")
        print(f"INSTANCES PER CLASS")
        print(f"{'='*60}")
        for class_id, count in sorted(class_counts.items()):
            print(f"Class {class_id}: {count} instance(s)")


def demonstrate_instance_extraction():
    """Full demonstration of instance extraction"""

    print("="*60)
    print("INSTANCE SEGMENTATION FROM SEMANTIC BELIEF")
    print("="*60)

    # Initialize
    print("\nInitializing pipeline...")
    mem = ManipulationEnhancedMapping(render=False, show_vis=False)
    mem.reset_env()

    # Get initial observation
    print("Getting observations...")
    cam_data, gt_data = mem.get_processed_array_and_gt_data()
    height_hms = np.array(cam_data['height_maps'])
    semantic_hms = np.array(cam_data['semantic_maps'])
    invalid_mask = height_hms[..., 0] == 0
    semantic_hms[invalid_mask] = mem.n_classes

    # Initialize belief
    print("Initializing belief...")
    previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
        torch.ones((1, 1, 204, 120, 200), device='cuda'))

    # Do a few observations to build up the belief
    print("\nCollecting observations to build belief...")
    from shelf_gym.utils.model_evaluation_utils import get_igs_for_map

    previous_views = []
    for step in range(5):
        # Select view
        igs, _ = get_igs_for_map(previous_map, mem.ig_calc, skip=1, use_alternative=True)
        igs[previous_views] = 0
        viewpoint = int(igs.argmax())

        # Execute observation
        previous_map, previous_semantic_map = mem.execute_observation(
            previous_views, viewpoint, previous_map, previous_semantic_map)

        print(f"  Step {step+1}: observed from viewpoint {viewpoint}")

    # Extract instances
    print("\n" + "="*60)
    print("EXTRACTING INSTANCES...")
    print("="*60)

    extractor = InstanceExtractor(mem)

    # Extract with different thresholds
    output_dir = './instance_extraction'
    os.makedirs(output_dir, exist_ok=True)

    for conf_thresh in [0.2, 0.5, 0.8]:
        print(f"\n--- Confidence Threshold: {conf_thresh} ---")

        instance_map, instance_info = extractor.extract_instances(
            previous_semantic_map,
            confidence_threshold=conf_thresh,
            min_pixels=10
        )

        # Print summary
        extractor.print_instance_summary(instance_info)

        # Visualize
        save_path = os.path.join(output_dir, f'instances_conf_{conf_thresh}.png')
        extractor.visualize_instances(instance_map, instance_info,
                                     previous_semantic_map, save_path)

    mem.close()

    print(f"\n{'='*60}")
    print(f"Done! Instance segmentations saved to: {output_dir}/")
    print(f"{'='*60}")

    return extractor, instance_map, instance_info


if __name__ == '__main__':
    extractor, instance_map, instance_info = demonstrate_instance_extraction()
