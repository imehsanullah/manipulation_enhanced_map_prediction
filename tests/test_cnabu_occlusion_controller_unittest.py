from __future__ import annotations

import unittest

import numpy as np

from scene_graph_mem.runtime.cnabu_scene_graph import encode_binary_mask_rle
from shelf_gym.utils.cnabu_occlusion_planner import (
    BeliefOcclusionAllocationController,
)


class _FakeNodeSplitter:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.calls = []

    def predict_nodes(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "schema": "mem_cnabu_learned_component_splitter_nodes_v1",
            "nodes": [
                {
                    "id": 1,
                    "class_id": 2,
                    "mask": encode_binary_mask_rle(self.mask),
                }
            ],
            "metadata": {
                "node_source": "learned_component_splitter",
                "physical_relation_executed": False,
                "physical_relation_assets_or_records_loaded": False,
            },
        }


class _FakeInfoGain:
    def __init__(self) -> None:
        self.calls = []

    def get_raycast(self, *, camera_idx):
        self.calls.append(int(camera_idx))
        # raw x=5 maps to canonical x=1 for a width-six MEM map.
        return np.asarray([[[5, 1, 0], [5, 1, 1]]], dtype=np.uint8)


class _ForbiddenGTEnvironment:
    def get_gt_height_map(self, **kwargs):
        raise AssertionError("deterministic controller attempted to read GT")


class BeliefOcclusionControllerTest(unittest.TestCase):
    def test_deterministic_controller_is_gt_free_and_node_only(self) -> None:
        alpha = np.ones((10, 6, 6), dtype=np.float32)
        beta = np.full((10, 6, 6), 9.0, dtype=np.float32)
        alpha[:, 1:4, 1:3] = 9.0
        beta[:, 1:4, 1:3] = 1.0
        distribution = np.empty((1, 20, 6, 6), dtype=np.float32)
        distribution[0, 0::2] = beta
        distribution[0, 1::2] = alpha
        semantics = np.ones((1, 15, 6, 6), dtype=np.float32)
        semantics[:, 14] = 20.0
        semantics[:, 14, 1:4, 1:3] = 1.0
        semantics[:, 2, 1:4, 1:3] = 20.0
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:4, 1:3] = True
        splitter = _FakeNodeSplitter(mask)
        info_gain = _FakeInfoGain()
        controller = BeliefOcclusionAllocationController(
            arm="deterministic",
            node_splitter=splitter,
            info_gain=info_gain,
            device="cpu",
            raw_shape_hw=(6, 6),
            crop_rows=(0, 6),
            source_batch_size=1,
        )

        allocator = controller.build_allocator(
            occupancy_distribution=distribution,
            semantic_concentration=semantics,
            camera_index=7,
            environment=_ForbiddenGTEnvironment(),
        )

        self.assertEqual(info_gain.calls, [7])
        self.assertEqual(len(splitter.calls), 1)
        self.assertEqual(allocator.source_masks_raw_hw.shape, (1, 6, 6))
        self.assertEqual(len(controller.history), 1)
        record = controller.history[0]
        self.assertFalse(record["uses_gt"])
        self.assertTrue(record["deployable"])
        self.assertFalse(record["physical_relation_executed"])
        self.assertFalse(record["physical_relation_assets_or_records_loaded"])
        self.assertEqual(
            record["cache"]["support_provenance"],
            "runtime_cnabu_learned_component_splitter",
        )

        controller.record_sampling_result(
            allocation_diagnostics={
                "best_source_index": 0,
                "best_source_has_feasible_path": True,
                "feasible_path_source_indices": [0],
            },
            feasible_path_count=1,
        )
        controller.record_scoring_result(selected_candidate_index=0)
        controller.record_execution_result(
            executed=True,
            push_return_code=0,
            object_drop=False,
        )
        self.assertEqual(record["selected_candidate"]["source_index"], 0)
        self.assertTrue(
            record["selected_candidate"]["source_is_best_attribution_source"]
        )
        self.assertTrue(record["execution"]["accepted_without_tilt_or_drop"])


if __name__ == "__main__":
    unittest.main()
