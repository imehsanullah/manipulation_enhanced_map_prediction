from __future__ import annotations

import unittest

import numpy as np

from shelf_gym.scripts.run_cnabu_scene_graph_live_demo import (
    DEFAULT_SHELF_VIEW_XYXY,
    _align_gt_to_raw_view,
    build_gt_instance_scene_graph,
    cnabu_mean_arrays_from_live_belief,
    compose_graph_gt_panel,
    render_gt_topdown_panel,
)


class CnabuSceneGraphLiveVizTest(unittest.TestCase):
    def test_live_belief_means_preserve_interleaved_cnabu_contract(self) -> None:
        occupancy = np.stack(
            [
                np.full((2, 3), 3.0),
                np.full((2, 3), 1.0),
                np.full((2, 3), 1.0),
                np.full((2, 3), 3.0),
            ],
            axis=0,
        )[None, ...]
        semantic = np.stack(
            [np.full((2, 3), 1.0), np.full((2, 3), 3.0)], axis=0
        )[None, ...]

        occupancy_mean, semantic_mean = cnabu_mean_arrays_from_live_belief(
            occupancy, semantic
        )

        self.assertEqual(occupancy_mean.shape, (2, 2, 3))
        self.assertEqual(semantic_mean.shape, (2, 2, 3))
        np.testing.assert_allclose(occupancy_mean[0], 0.25)
        np.testing.assert_allclose(occupancy_mean[1], 0.75)
        np.testing.assert_allclose(semantic_mean[0], 0.25)
        np.testing.assert_allclose(semantic_mean[1], 0.75)

    def _gt_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        semantic = np.full((84, 158), 14, dtype=np.int32)
        occupancy = np.zeros((84, 158, 3), dtype=np.float32)
        semantic[10:20, 20:30] = 3
        occupancy[10:20, 20:30, :] = 1.0
        semantic[40:50, 22:32] = 4
        occupancy[40:50, 22:32, :] = 1.0
        return semantic, occupancy

    def _gt_data(self) -> dict[str, np.ndarray]:
        semantic, occupancy = self._gt_arrays()
        semantic_raw = np.full((140, 200), 14, dtype=np.int32)
        semantic_raw[35:119, 21:179] = semantic
        instance_maps = np.zeros((3, 140, 200), dtype=np.float32)
        instance_maps[0, 45:55, 41:51] = 101
        instance_maps[1, 75:85, 43:53] = 202
        instance_maps[2, 30:120, 20:180] = 999  # Non-object simulator body.
        return {
            "semantic_gt": semantic,
            "semantic_gt_raw": semantic_raw,
            "voxel_height_map": occupancy,
            "instance_maps": instance_maps,
            "object_instance_ids": np.asarray([101, 202], dtype=np.int64),
            "object_class_ids": np.asarray([3, 4], dtype=np.int64),
        }

    def test_gt_crop_is_aligned_to_the_fixed_mem_shelf_view(self) -> None:
        semantic, occupancy = self._gt_arrays()

        labels, occupied, valid = _align_gt_to_raw_view(
            semantic,
            occupancy.max(axis=2),
            view_xyxy=DEFAULT_SHELF_VIEW_XYXY,
        )

        self.assertEqual(labels.shape, (94, 176))
        self.assertEqual(occupied.shape, (94, 176))
        self.assertEqual(valid.shape, (94, 176))
        self.assertEqual(int(labels[13, 29]), 3)
        self.assertEqual(float(occupied[13, 29]), 1.0)
        self.assertTrue(bool(valid[3, 9]))
        self.assertFalse(bool(valid[0, 0]))

        full_labels, full_occupied, full_valid = _align_gt_to_raw_view(
            semantic,
            occupancy.max(axis=2),
            view_xyxy=None,
        )
        self.assertEqual(full_labels.shape, (140, 200))
        self.assertEqual(int(full_labels[45, 41]), 3)
        self.assertEqual(float(full_occupied[45, 41]), 1.0)
        self.assertTrue(bool(full_valid[35, 21]))

    def test_gt_panel_matches_prediction_dimensions_and_composes_side_by_side(self) -> None:
        gt_panel = render_gt_topdown_panel(
            self._gt_data(),
            width=1000,
            height=620,
            update_index=6,
            view_xyxy=DEFAULT_SHELF_VIEW_XYXY,
            rotate_180=True,
        )
        prediction_panel = np.full_like(gt_panel, 248)

        comparison = compose_graph_gt_panel(prediction_panel, gt_panel)

        self.assertEqual(gt_panel.shape, (620, 1000, 3))
        self.assertEqual(gt_panel.dtype, np.uint8)
        self.assertGreater(float(gt_panel.std()), 8.0)
        self.assertEqual(comparison.shape, (620, 2016, 3))

    def test_gt_instance_masks_build_evaluation_nodes_and_rule_edges(self) -> None:
        graph, context = build_gt_instance_scene_graph(self._gt_data())

        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(
            [node["simulator_instance_id"] for node in graph["nodes"]],
            [101, 202],
        )
        self.assertEqual(graph["edges"][0]["source"], 1)
        self.assertEqual(graph["edges"][0]["target"], 2)
        self.assertTrue(graph["metadata"]["evaluation_only"])
        self.assertTrue(graph["metadata"]["uses_simulator_instance_labels"])
        self.assertEqual(context["background_bgr"].shape, (140, 200, 3))


if __name__ == "__main__":
    unittest.main()
