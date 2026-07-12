from __future__ import annotations

import json
import unittest

import numpy as np

from shelf_gym.utils.cnabu_scene_graph import predict_scene_graph_from_cnabu
from shelf_gym.utils.cnabu_scene_graph_viz import (
    build_cnabu_map_context,
    compose_runtime_demo_panel,
    render_cnabu_belief_map_view,
    render_cnabu_scene_graph_view,
)


class CnabuSceneGraphVizTest(unittest.TestCase):
    def _belief_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        height, width = 64, 80
        occupancy_mean = np.zeros((3, height, width), dtype=np.float32)
        semantic_mean = np.zeros((15, height, width), dtype=np.float32)
        semantic_mean[14, :, :] = 1.0

        def add_box(class_id: int, y1: int, y2: int, x1: int, x2: int) -> None:
            occupancy_mean[:, y1:y2, x1:x2] = 0.92
            semantic_mean[14, y1:y2, x1:x2] = 0.05
            semantic_mean[class_id, y1:y2, x1:x2] = 0.95

        add_box(2, 10, 20, 20, 40)
        add_box(2, 40, 50, 25, 45)
        add_box(6, 15, 27, 55, 72)
        return occupancy_mean, semantic_mean

    def test_render_graph_view_uses_context_and_preserves_json_safe_graph(self) -> None:
        occupancy_mean, semantic_mean = self._belief_arrays()
        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )
        context = build_cnabu_map_context(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )

        image = render_cnabu_scene_graph_view(
            graph,
            context=context,
            update_index=2,
            width=640,
            height=480,
            max_edges=8,
            max_labels=4,
            show_context_background=True,
        )

        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image.std()), 5.0)
        self.assertEqual(context["background_bgr"].shape, (64, 80, 3))
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 1)
        json.dumps(graph)

    def test_render_graph_view_handles_crop_rows_without_changing_graph_shape(self) -> None:
        occupancy_mean, semantic_mean = self._belief_arrays()
        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(84, 80),
            crop_rows=(10, 74),
            include_masks=True,
        )
        context = build_cnabu_map_context(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(84, 80),
            crop_rows=(10, 74),
        )

        image = render_cnabu_scene_graph_view(graph, context=context, width=640, height=480)

        self.assertEqual(image.shape, (480, 640, 3))
        self.assertGreater(float(image.std()), 5.0)
        self.assertEqual(context["background_bgr"].shape, (84, 80, 3))
        self.assertEqual(graph["metadata"]["raw_shape_hw"], [84, 80])
        self.assertEqual(graph["metadata"]["crop_rows"], [10, 74])
        self.assertEqual(graph["nodes"][0]["mask"]["size"], [84, 80])

    def test_render_belief_map_view_uses_context_without_graph_payload(self) -> None:
        occupancy_mean, semantic_mean = self._belief_arrays()
        context = build_cnabu_map_context(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )

        image = render_cnabu_belief_map_view(
            context=context,
            update_index=3,
            width=640,
            height=480,
        )

        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image.std()), 5.0)
        self.assertEqual(context["background_bgr"].shape, (64, 80, 3))

    def test_render_topdown_views_can_rotate_without_mutating_graph(self) -> None:
        occupancy_mean, semantic_mean = self._belief_arrays()
        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
            include_masks=True,
        )
        graph_before = json.dumps(graph, sort_keys=True)
        context = build_cnabu_map_context(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )

        graph_image = render_cnabu_scene_graph_view(
            graph,
            context=context,
            width=640,
            height=480,
            rotate_map_180=True,
        )
        belief_image = render_cnabu_belief_map_view(
            context=context,
            width=640,
            height=480,
            rotate_map_180=True,
        )

        self.assertEqual(graph_image.shape, (480, 640, 3))
        self.assertEqual(belief_image.shape, (480, 640, 3))
        self.assertGreater(float(graph_image.std()), 5.0)
        self.assertGreater(float(belief_image.std()), 5.0)
        self.assertEqual(json.dumps(graph, sort_keys=True), graph_before)

    def test_runtime_panel_composition_accepts_rgb_scene_and_bgr_graph(self) -> None:
        occupancy_mean, semantic_mean = self._belief_arrays()
        graph = predict_scene_graph_from_cnabu(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )
        context = build_cnabu_map_context(
            occupancy_mean=occupancy_mean,
            semantic_mean=semantic_mean,
            raw_shape_hw=(64, 80),
            crop_rows=(0, 64),
        )
        graph_image = render_cnabu_scene_graph_view(graph, context=context, width=640, height=480)
        scene_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        scene_rgb[:, :, 0] = 80
        scene_rgb[60:190, 90:230, 1] = 160
        scene_rgb[90:150, 130:190, 2] = 240

        panel = compose_runtime_demo_panel(scene_rgb, graph_image, width=1000, height=520)

        self.assertEqual(panel.shape, (520, 1000, 3))
        self.assertEqual(panel.dtype, np.uint8)
        self.assertGreater(float(panel.std()), 5.0)


if __name__ == "__main__":
    unittest.main()
