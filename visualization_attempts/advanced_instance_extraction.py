"""
Advanced Instance Segmentation Methods
Implements multiple sophisticated approaches to extract instances from semantic beliefs
"""

import matplotlib
matplotlib.use('Agg')

from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
from shelf_gym.utils.model_evaluation_utils import get_igs_for_map
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from skimage import measure, morphology, segmentation, feature
from skimage.filters import gaussian
from sklearn.cluster import DBSCAN, MeanShift
import cv2
import os


class AdvancedInstanceExtractor:
    """Multiple sophisticated instance extraction methods"""

    def __init__(self, mem_pipeline):
        self.mem = mem_pipeline
        self.n_classes = mem_pipeline.n_classes

    def preprocess_belief(self, semantic_map, confidence_threshold=0.5):
        """Common preprocessing for all methods"""
        sem_color, _, sem_labels, sem_conf = self.mem.get_semantic_rgb_image(semantic_map)

        # Filter by confidence
        sem_labels_filtered = np.where(sem_conf >= confidence_threshold, sem_labels, -1)

        return sem_labels_filtered, sem_conf, sem_color

    # ==================== METHOD 1: Distance Transform + Watershed ====================
    def method1_distance_watershed(self, semantic_map, confidence_threshold=0.5, min_pixels=10):
        """
        Method 1: Distance Transform + Watershed
        - Computes distance transform for each class
        - Finds local maxima as markers
        - Applies watershed segmentation
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Distance transform
            distance = ndimage.distance_transform_edt(class_mask)

            # Find local maxima as markers
            local_maxima = morphology.local_maxima(distance, indices=False)
            markers, num_markers = ndimage.label(local_maxima)

            if num_markers == 0:
                continue

            # Watershed segmentation
            labels = segmentation.watershed(-distance, markers, mask=class_mask)

            # Process each watershed region
            for region_id in range(1, labels.max() + 1):
                region_mask = (labels == region_id)
                if region_mask.sum() < min_pixels:
                    continue

                instance_map[region_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, region_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== METHOD 2: Marker-Controlled Watershed ====================
    def method2_marker_watershed(self, semantic_map, confidence_threshold=0.5, min_pixels=10):
        """
        Method 2: Marker-Controlled Watershed with Confidence
        - Uses high-confidence regions as sure foreground markers
        - Uses medium-confidence as uncertain regions
        - Applies watershed with markers
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Create confidence-based markers
            sure_fg = (class_mask) & (sem_conf > 0.85)  # High confidence
            unknown = (class_mask) & (sem_conf > confidence_threshold) & (sem_conf <= 0.85)

            if not sure_fg.any():
                sure_fg = class_mask

            # Morphological operations to clean markers
            sure_fg = morphology.binary_erosion(sure_fg, morphology.disk(2))
            sure_fg = morphology.binary_dilation(sure_fg, morphology.disk(1))

            # Label sure foreground regions as markers
            markers, num_markers = ndimage.label(sure_fg)

            if num_markers == 0:
                continue

            # Distance transform for watershed
            distance = ndimage.distance_transform_edt(class_mask)

            # Watershed
            labels = segmentation.watershed(-distance, markers, mask=class_mask)

            # Process regions
            for region_id in range(1, labels.max() + 1):
                region_mask = (labels == region_id)
                if region_mask.sum() < min_pixels:
                    continue

                instance_map[region_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, region_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== METHOD 3: Contour-Based Segmentation ====================
    def method3_contour_based(self, semantic_map, confidence_threshold=0.5, min_pixels=10):
        """
        Method 3: Contour-Based Segmentation
        - Finds contours for each class
        - Filters contours by area and hierarchy
        - Handles nested contours
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id).astype(np.uint8)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Morphological closing to fill small gaps
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

            # Find contours
            contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < min_pixels:
                    continue

                # Create mask for this contour
                contour_mask = np.zeros_like(class_mask, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 1, -1)
                contour_mask = contour_mask.astype(bool)

                instance_map[contour_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, contour_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== METHOD 4: DBSCAN Clustering ====================
    def method4_dbscan_clustering(self, semantic_map, confidence_threshold=0.5,
                                  min_pixels=10, eps=5, min_samples=5):
        """
        Method 4: DBSCAN Spatial Clustering
        - Uses DBSCAN on pixel coordinates
        - Handles irregular shapes well
        - Automatically determines number of clusters
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Get coordinates of pixels in this class
            coords = np.column_stack(np.where(class_mask))

            if len(coords) < min_samples:
                # Too few points for DBSCAN, treat as one instance
                instance_map[class_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, class_mask, sem_conf))
                next_id += 1
                continue

            # Apply DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_

            # Process each cluster (excluding noise: label -1)
            for cluster_id in set(labels):
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_coords = coords[labels == cluster_id]
                cluster_mask = np.zeros_like(class_mask)
                cluster_mask[cluster_coords[:, 0], cluster_coords[:, 1]] = True

                if cluster_mask.sum() < min_pixels:
                    continue

                instance_map[cluster_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, cluster_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== METHOD 5: MeanShift Clustering ====================
    def method5_meanshift_clustering(self, semantic_map, confidence_threshold=0.5,
                                    min_pixels=10, bandwidth=10):
        """
        Method 5: MeanShift Clustering
        - Uses MeanShift on spatial coordinates
        - Good for finding natural groupings
        - Bandwidth controls granularity
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Get coordinates
            coords = np.column_stack(np.where(class_mask)).astype(float)

            if len(coords) < 3:
                # Too few points
                instance_map[class_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, class_mask, sem_conf))
                next_id += 1
                continue

            # Apply MeanShift
            clustering = MeanShift(bandwidth=bandwidth).fit(coords)
            labels = clustering.labels_

            # Process each cluster
            for cluster_id in set(labels):
                cluster_coords = coords[labels == cluster_id].astype(int)
                cluster_mask = np.zeros_like(class_mask)
                cluster_mask[cluster_coords[:, 0], cluster_coords[:, 1]] = True

                if cluster_mask.sum() < min_pixels:
                    continue

                instance_map[cluster_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, cluster_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== METHOD 6: Hybrid Confidence-Watershed ====================
    def method6_hybrid_confidence_watershed(self, semantic_map, confidence_threshold=0.5,
                                           min_pixels=10):
        """
        Method 6: Hybrid Confidence-Weighted Watershed
        - Combines confidence and spatial information
        - Uses confidence gradient for watershed
        - More robust to noise
        """
        sem_labels, sem_conf, _ = self.preprocess_belief(semantic_map, confidence_threshold)

        instance_map = np.zeros_like(sem_labels, dtype=np.int32)
        instance_info = []
        next_id = 1

        for class_id in range(self.n_classes):
            class_mask = (sem_labels == class_id)
            if not class_mask.any() or class_mask.sum() < min_pixels:
                continue

            # Create confidence-weighted distance
            conf_in_class = sem_conf.copy()
            conf_in_class[~class_mask] = 0

            # Smooth confidence
            conf_smooth = gaussian(conf_in_class, sigma=2)

            # Distance transform
            distance = ndimage.distance_transform_edt(class_mask)

            # Combine distance and confidence
            weighted_distance = distance * (conf_smooth + 0.1)  # Add small epsilon

            # Find peaks in weighted distance
            peaks = morphology.local_maxima(weighted_distance, indices=False)
            markers, num_markers = ndimage.label(peaks)

            if num_markers == 0:
                continue

            # Watershed on negative weighted distance
            labels = segmentation.watershed(-weighted_distance, markers, mask=class_mask)

            # Process regions
            for region_id in range(1, labels.max() + 1):
                region_mask = (labels == region_id)
                if region_mask.sum() < min_pixels:
                    continue

                instance_map[region_mask] = next_id
                instance_info.append(self._get_instance_info(
                    next_id, class_id, region_mask, sem_conf))
                next_id += 1

        return instance_map, instance_info

    # ==================== Helper Functions ====================
    def _get_instance_info(self, instance_id, class_id, mask, sem_conf):
        """Extract metadata for an instance"""
        rows, cols = np.where(mask)
        bbox = [cols.min(), rows.min(), cols.max(), rows.max()]
        instance_conf = sem_conf[mask].mean()

        return {
            'instance_id': instance_id,
            'class_id': int(class_id),
            'num_pixels': int(mask.sum()),
            'bbox': bbox,
            'confidence': float(instance_conf),
            'centroid': [float(cols.mean()), float(rows.mean())]
        }

    # ==================== Comparison & Visualization ====================
    def compare_all_methods(self, semantic_map, confidence_threshold=0.5, min_pixels=10):
        """Run all methods and return results"""
        methods = {
            'Distance Watershed': self.method1_distance_watershed,
            'Marker Watershed': self.method2_marker_watershed,
            'Contour-Based': self.method3_contour_based,
            'DBSCAN': self.method4_dbscan_clustering,
            'MeanShift': self.method5_meanshift_clustering,
            'Hybrid Confidence': self.method6_hybrid_confidence_watershed,
        }

        results = {}
        for name, method in methods.items():
            print(f"\nRunning: {name}...")
            try:
                instance_map, instance_info = method(semantic_map, confidence_threshold, min_pixels)
                results[name] = {
                    'instance_map': instance_map,
                    'instance_info': instance_info,
                    'num_instances': len(instance_info)
                }
                print(f"  ✓ Detected {len(instance_info)} instances")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[name] = None

        return results

    def visualize_comparison(self, results, semantic_map, save_path='comparison.png'):
        """Create comprehensive comparison visualization"""
        # Filter out failed methods
        valid_results = {k: v for k, v in results.items() if v is not None}
        n_methods = len(valid_results)

        if n_methods == 0:
            print("No valid results to visualize")
            return

        # Get semantic visualization
        sem_color, _, _, sem_conf = self.mem.get_semantic_rgb_image(semantic_map)

        # Create figure
        n_rows = (n_methods + 2) // 2 + 1  # +1 for original, then methods in pairs
        fig = plt.figure(figsize=(20, 4 * n_rows))
        gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.2)

        # Original semantic
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(sem_color)
        ax_orig.set_title('Original Semantic Segmentation', fontsize=12, fontweight='bold')
        ax_orig.axis('off')

        # Confidence
        ax_conf = fig.add_subplot(gs[0, 1])
        im = ax_conf.imshow(sem_conf, cmap='hot', vmin=0, vmax=1)
        ax_conf.set_title('Confidence Map', fontsize=12, fontweight='bold')
        ax_conf.axis('off')
        plt.colorbar(im, ax=ax_conf, fraction=0.046)

        # Plot each method
        np.random.seed(42)
        for idx, (name, result) in enumerate(valid_results.items(), start=1):
            row = idx // 2 + 1
            col = idx % 2

            ax = fig.add_subplot(gs[row, col])

            instance_map = result['instance_map']
            num_instances = result['num_instances']

            # Color instances
            max_id = instance_map.max()
            if max_id > 0:
                colors = np.random.randint(50, 255, size=(max_id + 1, 3))
                colors[0] = [0, 0, 0]  # Background black
                instance_vis = colors[instance_map]
            else:
                instance_vis = np.zeros((*instance_map.shape, 3), dtype=np.uint8)

            ax.imshow(instance_vis)
            ax.set_title(f'{name}\n({num_instances} instances)',
                        fontsize=11, fontweight='bold')
            ax.axis('off')

        fig.suptitle('Instance Segmentation: Method Comparison',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison: {save_path}")

    def create_detailed_comparison(self, results, semantic_map, save_dir='./method_comparison'):
        """Create detailed individual visualizations for each method"""
        os.makedirs(save_dir, exist_ok=True)

        sem_color, _, _, sem_conf = self.mem.get_semantic_rgb_image(semantic_map)

        for name, result in results.items():
            if result is None:
                continue

            instance_map = result['instance_map']
            instance_info = result['instance_info']

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Original semantic
            axes[0].imshow(sem_color)
            axes[0].set_title('Semantic', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Instances with bounding boxes
            np.random.seed(42)
            max_id = instance_map.max()
            if max_id > 0:
                colors = np.random.randint(50, 255, size=(max_id + 1, 3))
                colors[0] = [0, 0, 0]
                instance_vis = colors[instance_map]
            else:
                instance_vis = np.zeros((*instance_map.shape, 3), dtype=np.uint8)

            axes[1].imshow(instance_vis)
            axes[1].set_title(f'Instances ({len(instance_info)} found)',
                            fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # Draw bounding boxes
            for info in instance_info:
                bbox = info['bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]),
                                    bbox[2] - bbox[0], bbox[3] - bbox[1],
                                    fill=False, edgecolor='red', linewidth=1.5)
                axes[1].add_patch(rect)

            # Instance map with IDs
            axes[2].imshow(instance_map, cmap='nipy_spectral', interpolation='nearest')
            axes[2].set_title('Instance IDs', fontsize=12, fontweight='bold')
            axes[2].axis('off')

            fig.suptitle(f'Method: {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            safe_name = name.replace(' ', '_').lower()
            save_path = os.path.join(save_dir, f'{safe_name}_detailed.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")


def run_comparison():
    """Run full comparison of all methods"""
    print("="*80)
    print("ADVANCED INSTANCE SEGMENTATION - METHOD COMPARISON")
    print("="*80)

    # Initialize
    print("\nInitializing...")
    mem = ManipulationEnhancedMapping(render=False, show_vis=False)
    mem.reset_env()

    # Build belief
    print("Building belief map with observations...")
    cam_data, gt_data = mem.get_processed_array_and_gt_data()
    height_hms = np.array(cam_data['height_maps'])
    semantic_hms = np.array(cam_data['semantic_maps'])
    invalid_mask = height_hms[..., 0] == 0
    semantic_hms[invalid_mask] = mem.n_classes

    previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
        torch.ones((1, 1, 204, 120, 200), device='cuda'))

    previous_views = []
    for step in range(5):
        igs, _ = get_igs_for_map(previous_map, mem.ig_calc, skip=1, use_alternative=True)
        igs[previous_views] = 0
        viewpoint = int(igs.argmax())
        previous_map, previous_semantic_map = mem.execute_observation(
            previous_views, viewpoint, previous_map, previous_semantic_map)
        print(f"  Step {step+1}/5 complete")

    # Run comparison
    print("\n" + "="*80)
    print("RUNNING ALL METHODS")
    print("="*80)

    extractor = AdvancedInstanceExtractor(mem)
    results = extractor.compare_all_methods(previous_semantic_map,
                                           confidence_threshold=0.5,
                                           min_pixels=10)

    # Print statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Instances':<12} {'Status'}")
    print("-"*80)
    for name, result in results.items():
        if result:
            print(f"{name:<25} {result['num_instances']:<12} ✓ Success")
        else:
            print(f"{name:<25} {'N/A':<12} ✗ Failed")

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Get script directory for output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'advanced_instance_comparison')
    os.makedirs(output_dir, exist_ok=True)

    # Main comparison
    extractor.visualize_comparison(results, previous_semantic_map,
                                  save_path=os.path.join(output_dir, 'all_methods_comparison.png'))

    # Detailed views
    extractor.create_detailed_comparison(results, previous_semantic_map,
                                        save_dir=output_dir)

    mem.close()

    print("\n" + "="*80)
    print(f"COMPLETE! All results saved to: {output_dir}/")
    print("="*80)

    return extractor, results


if __name__ == '__main__':
    extractor, results = run_comparison()
