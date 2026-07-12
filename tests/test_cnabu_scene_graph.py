from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shelf_gym.utils.cnabu_scene_graph import (
    decode_binary_mask_rle,
    predict_scene_graph_from_cnabu,
)


class CnabuSceneGraphTest(unittest.TestCase):
    def _mean_belief_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = 64, 80
        occupancy_mean = np.zeros((2, height, width), dtype=np.float32)
        semantic_mean = np.zeros((15, height, width), dtype=np.float32)
        semantic_mean[14, :, :] = 1.0
        occupancy_epistemic = np.full_like(occupancy_mean, 0.01)
        semantic_vacuity = np.full((height, width), 0.2, dtype=np.float32)

        def add_box(class_id: int, y1: int, y2: int, x1: int, x2: int) -> None:
            occupancy_mean[:, y1:y2, x1:x2] = 0.90
            semantic_mean[14, y1:y2, x1:x2] = 0.10
            semantic_mean[class_id, y1:y2, x1:x2] = 0.90
            semantic_vacuity[y1:y2, x1:x2] = 0.05

        add_box(2, 10, 20, 20, 40)
        add_box(2, 40, 50, 25, 45)
        add_box(2, 12, 22, 60, 72)
        return occupancy_mean, semantic_mean, occupancy_epistemic, semantic_vacuity

    def _merged_same_class_bridge_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 48, 48
        occupancy_mean = np.zeros((4, height, width), dtype=np.float32)
        semantic_mean = np.zeros((15, height, width), dtype=np.float32)
        semantic_mean[14, :, :] = 1.0

        class_id = 4

        def set_semantics(y1: int, y2: int, x1: int, x2: int) -> None:
            semantic_mean[14, y1:y2, x1:x2] = 0.05
            semantic_mean[class_id, y1:y2, x1:x2] = 0.95

        occupancy_mean[1:3, 8:18, 18:30] = 0.92
        occupancy_mean[1:3, 18:28, 22:26] = 0.55
        occupancy_mean[1:3, 28:38, 18:30] = 0.92
        set_semantics(8, 18, 18, 30)
        set_semantics(18, 28, 22, 26)
        set_semantics(28, 38, 18, 30)
        return occupancy_mean, semantic_mean

    def _large_single_lobe_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 48, 48
        occupancy_mean = np.zeros((3, height, width), dtype=np.float32)
        semantic_mean = np.zeros((15, height, width), dtype=np.float32)
        semantic_mean[14, :, :] = 1.0
        class_id = 4
        occupancy_mean[:, 10:38, 12:36] = 0.92
        semantic_mean[14, 10:38, 12:36] = 0.05
        semantic_mean[class_id, 10:38, 12:36] = 0.95
        return occupancy_mean, semantic_mean

    def test_predict_scene_graph_from_mean_arrays_is_json_safe_and_directed(self) -> None:
        occupancy_mean, semantic_mean, occupancy_epistemic, semantic_vacuity = self._mean_belief_arrays()

        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            occupancy_epistemic=occupancy_epistemic,
            semantic_vacuity=semantic_vacuity,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )

        self.assertEqual(graph["schema"], "mem_cnabu_rule_scene_graph_v0")
        self.assertFalse(graph["metadata"]["requires_gt"])
        self.assertFalse(graph["metadata"]["uses_d3g"])
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 1)
        edge = graph["edges"][0]
        self.assertEqual(edge["predicate"], "blocks_access_to")
        self.assertLess(
            graph["nodes"][edge["source_index"]]["centroid_yx"][0],
            graph["nodes"][edge["target_index"]]["centroid_yx"][0],
        )
        self.assertGreaterEqual(edge["lateral_overlap_union"], 0.5)
        self.assertEqual(graph["adjacency_matrix"][edge["source_index"]][edge["target_index"]], 1)

        first_node = graph["nodes"][0]
        self.assertIn("mask", first_node)
        self.assertEqual(decode_binary_mask_rle(first_node["mask"]).shape, (64, 80))
        self.assertIn("mean_occupancy", first_node["confidence"])
        self.assertIn("mean_occupancy_epistemic", first_node["uncertainty"])
        json.dumps(graph)

    def test_predict_scene_graph_from_npz_derives_evidential_means_and_uncertainty(self) -> None:
        occupancy_mean, semantic_mean, _, _ = self._mean_belief_arrays()
        occupancy_alpha = 1.0 + occupancy_mean * 20.0
        occupancy_beta = 1.0 + (1.0 - occupancy_mean) * 20.0
        semantic_concentration = 1.0 + semantic_mean * 20.0

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cnabu_hms.npz"
            np.savez_compressed(
                path,
                occupancy_alpha=occupancy_alpha.astype(np.float32),
                occupancy_beta=occupancy_beta.astype(np.float32),
                semantic_concentration=semantic_concentration.astype(np.float32),
                selected_view_indices=np.asarray([0, 1, 2], dtype=np.int16),
                crop_rows=np.asarray([0, 64], dtype=np.int16),
                raw_shape_hms=np.asarray([300, 64, 80, 2], dtype=np.int32),
                cnabu_shape_hw=np.asarray([64, 80], dtype=np.int32),
                metadata_json=np.asarray(json.dumps({"sample_id": "synthetic"})),
            )

            graph = predict_scene_graph_from_cnabu(cnabu_path=path)

        self.assertEqual(graph["metadata"]["source"], "cnabu_hms_npz")
        self.assertEqual(graph["metadata"]["selected_view_indices"], [0, 1, 2])
        self.assertEqual(graph["metadata"]["source_metadata"]["sample_id"], "synthetic")
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertIsNotNone(graph["nodes"][0]["uncertainty"]["mean_occupancy_epistemic"])
        self.assertIsNotNone(graph["nodes"][0]["uncertainty"]["mean_semantic_vacuity"])
        json.dumps(graph)

    def test_runtime_interleaved_occupancy_distribution_is_supported(self) -> None:
        occupancy_mean, semantic_mean, _, _ = self._mean_belief_arrays()
        alpha = 1.0 + occupancy_mean * 20.0
        beta = 1.0 + (1.0 - occupancy_mean) * 20.0
        distribution = np.empty((1, alpha.shape[0] * 2, alpha.shape[1], alpha.shape[2]), dtype=np.float32)
        distribution[0, 0::2] = beta
        distribution[0, 1::2] = alpha
        semantic_concentration = (1.0 + semantic_mean * 20.0)[None]

        graph = predict_scene_graph_from_cnabu(
            occupancy_distribution=distribution,
            semantic_concentration=semantic_concentration,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )

        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["thresholds"]["edge_rule"]["access_axis"], "y")
        self.assertEqual(graph["thresholds"]["edge_rule"]["opening_side"], "low")
        json.dumps(graph)

    def test_optional_seeded_split_separates_weakly_merged_same_class_component(self) -> None:
        occupancy_mean, semantic_mean = self._merged_same_class_bridge_arrays()
        component_config = {
            "occupancy_threshold": 0.50,
            "min_voxels": 20,
            "min_pixels": 4,
            "connectivity": 1,
        }

        no_split_graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(48, 48),
            crop_rows=(0, 48),
            component_config=component_config,
            component_split_config={"enabled": False},
        )
        split_graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(48, 48),
            crop_rows=(0, 48),
            component_config=component_config,
            component_split_config={
                "enabled": True,
                "core_occupancy_threshold": 0.70,
                "min_seed_voxels": 20,
                "min_seed_pixels": 8,
                "min_split_voxels": 20,
                "min_split_pixels": 4,
            },
        )

        self.assertFalse(no_split_graph["thresholds"]["component_splitting"]["enabled"])
        self.assertEqual(len(no_split_graph["nodes"]), 1)
        self.assertFalse(no_split_graph["nodes"][0]["was_split"])
        self.assertEqual(no_split_graph["nodes"][0]["split"]["split_reason"], "disabled")

        self.assertTrue(split_graph["thresholds"]["component_splitting"]["enabled"])
        self.assertEqual(len(split_graph["nodes"]), 2)
        self.assertEqual(len(split_graph["edges"]), 1)
        self.assertEqual(
            split_graph["metadata"]["shape_info"]["component_splitting"]["num_split_parent_components"],
            1,
        )
        self.assertEqual(
            split_graph["metadata"]["shape_info"]["component_splitting"]["num_split_nodes"],
            2,
        )

        parent_ids = {node["split"]["parent_component_id"] for node in split_graph["nodes"]}
        self.assertEqual(len(parent_ids), 1)
        self.assertEqual({node["split"]["split_id"] for node in split_graph["nodes"]}, {1, 2})
        for node in split_graph["nodes"]:
            self.assertTrue(node["was_split"])
            self.assertEqual(node["split_method"], "seeded_distance_watershed")
            self.assertEqual(node["split"]["num_splits"], 2)
            self.assertEqual(node["split"]["core_seed_count"], 2)
            self.assertEqual(node["split"]["occupancy_threshold"], 0.50)
            self.assertEqual(node["split"]["core_occupancy_threshold"], 0.70)
            self.assertIn("parent_mean_occupancy", node["split"]["confidence"])
            self.assertIn("region_mean_occupancy", node["split"]["confidence"])

        edge = split_graph["edges"][0]
        self.assertLess(
            split_graph["nodes"][edge["source_index"]]["centroid_yx"][0],
            split_graph["nodes"][edge["target_index"]]["centroid_yx"][0],
        )
        json.dumps(no_split_graph)
        json.dumps(split_graph)

    def test_candidate_gated_2d_split_separates_footprint_lobes(self) -> None:
        occupancy_mean, semantic_mean = self._merged_same_class_bridge_arrays()
        prior_area = [100] * 14
        prior_width = [10] * 14
        prior_height = [10] * 14

        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(48, 48),
            crop_rows=(0, 48),
            component_config={
                "occupancy_threshold": 0.50,
                "min_voxels": 20,
                "min_pixels": 4,
                "connectivity": 1,
            },
            component_split_config={
                "enabled": True,
                "method": "candidate_gated_2d_footprint",
                "core_occupancy_threshold": 0.70,
                "min_seed_pixels": 8,
                "min_split_voxels": 20,
                "min_split_pixels": 4,
                "candidate_area_multiplier": 1.0,
                "candidate_bbox_multiplier": 1.0,
                "footprint_erosion_iterations": 2,
                "min_child_area_fraction": 0.20,
                "class_area_prior_pixels": prior_area,
                "class_width_prior_pixels": prior_width,
                "class_height_prior_pixels": prior_height,
            },
        )

        self.assertEqual(len(graph["nodes"]), 2)
        shape_info = graph["metadata"]["shape_info"]["component_splitting"]
        self.assertEqual(shape_info["num_split_candidate_components"], 1)
        self.assertEqual(shape_info["num_split_parent_components"], 1)
        self.assertEqual(shape_info["avg_children_per_split_component"], 2.0)
        for node in graph["nodes"]:
            self.assertTrue(node["was_split"])
            self.assertEqual(node["split_method"], "candidate_gated_2d_footprint")
            self.assertEqual(node["node_source"], "cnabu_2d_footprint_split_component")
            self.assertEqual(node["split"]["split_reason"], "candidate_2d_separated_lobes")
            self.assertTrue(node["split"]["candidate_considered"])
            self.assertIn("candidate_gate_reasons", node["split"])
            self.assertIn("class_size_prior", node["split"])
            self.assertGreater(node["split"]["split_confidence"], 0.0)
        json.dumps(graph)

    def test_candidate_gated_2d_split_refuses_single_lobe_candidate(self) -> None:
        occupancy_mean, semantic_mean = self._large_single_lobe_arrays()

        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(48, 48),
            crop_rows=(0, 48),
            component_config={
                "occupancy_threshold": 0.50,
                "min_voxels": 20,
                "min_pixels": 4,
                "connectivity": 1,
            },
            component_split_config={
                "enabled": True,
                "method": "candidate_gated_2d_footprint",
                "core_occupancy_threshold": 0.70,
                "min_seed_pixels": 8,
                "candidate_area_multiplier": 1.0,
                "candidate_bbox_multiplier": 1.0,
                "class_area_prior_pixels": [120] * 14,
                "class_width_prior_pixels": [10] * 14,
                "class_height_prior_pixels": [10] * 14,
            },
        )

        self.assertEqual(len(graph["nodes"]), 1)
        node = graph["nodes"][0]
        self.assertFalse(node["was_split"])
        self.assertTrue(node["split"]["candidate_considered"])
        self.assertEqual(node["split"]["split_reason"], "candidate_no_separated_2d_lobes")
        shape_info = graph["metadata"]["shape_info"]["component_splitting"]
        self.assertEqual(shape_info["num_split_candidate_components"], 1)
        self.assertEqual(shape_info["num_split_parent_components"], 0)
        self.assertEqual(shape_info["num_split_nodes"], 0)
        json.dumps(graph)


if __name__ == "__main__":
    unittest.main()
