from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scene_graph_mem.runtime.cnabu_scene_graph import encode_binary_mask_rle
from shelf_gym.scripts.psg_mem_x3_evaluator_adapter import (
    DEFAULT_FAILURE_DIAGNOSTICS,
    _failure_diagnostic,
    _oracle_rows,
    build_evaluator_adapter,
)
from shelf_gym.utils.psg_mem_live_registry import (
    LiveEpisodeHandle,
    close_live_episode,
    register_live_episode,
)


def _fixture() -> tuple[dict, np.ndarray]:
    predicted = np.zeros((12, 14), dtype=bool)
    predicted[2:6, 3:8] = True
    graph = {
        "nodes": [
            {
                "node_id": 7,
                "node_type": "object",
                "footprint_mask": encode_binary_mask_rle(predicted),
            }
        ]
    }
    instances = np.full(predicted.shape, -1, dtype=np.int64)
    instances[predicted] = 10
    instances[8:10, 9:12] = 20
    return graph, instances


class EvaluatorAdapterTest(unittest.TestCase):
    def tearDown(self) -> None:
        close_live_episode("evaluator-test")

    def test_physical_ids_are_translated_to_runtime_node_ids_only(self) -> None:
        graph, instances = _fixture()
        rows = _oracle_rows(
            graph,
            {
                "physical_blocker_instance_ids": [10],
                "blocker_candidate_counts": {"10": 3},
                "eligible_candidate_count": 6,
            },
            instances,
        )
        self.assertEqual(
            rows,
            [
                {
                    "source_node_id": 7,
                    "score": 0.5,
                    "evaluation_match_iou": 1.0,
                    "source": "physical_v1_evaluator_only",
                }
            ],
        )
        self.assertNotIn("10", repr(rows))

    def test_hidden_target_stays_inside_cached_evaluator_callbacks(self) -> None:
        graph, instances = _fixture()
        bridge = SimpleNamespace(
            state=SimpleNamespace(action_count=0, collision=False), history=[]
        )
        handle = LiveEpisodeHandle(
            episode_id="evaluator-test",
            mem=object(),
            bridge=bridge,
            scene_path="/fixed/scene.pkl",
            latest_graph=graph,
        )
        register_live_episode(handle)
        details = {
            "access_feasible": False,
            "candidate_count": 9,
            "eligible_candidate_count": 6,
            "clean_candidate_count": 0,
            "endpoint": "physical_clean_extraction_feasibility_read_only",
            "environment_state_restored": True,
            "_evaluation_private": {
                "physical_blocker_instance_ids": [10],
                "blocker_candidate_counts": {"10": 3},
                "eligible_candidate_count": 6,
            },
        }
        with patch(
            "shelf_gym.scripts.psg_mem_x3_evaluator_adapter.evaluate_live_target_access_feasibility",
            return_value=details,
        ) as evaluator, patch(
            "shelf_gym.scripts.psg_mem_x3_evaluator_adapter._current_instance_map",
            return_value=instances,
        ):
            callbacks = build_evaluator_adapter(
                {
                    "episode_id": "evaluator-test",
                    "arm_id": "h",
                    "evaluation_token": {"evaluation_object_id": 20},
                }
            )
            access = callbacks["access_evaluator"](
                evaluation_token={"evaluation_object_id": 20},
                actions_taken=0,
                graph_step=None,
            )
            rows = callbacks["oracle_blocker_provider"](
                episode_id="evaluator-test", step=0
            )

        self.assertIs(access["access_feasible"], False)
        self.assertIs(access["target_ever_visible"], True)
        self.assertIs(access["environment_state_restored"], True)
        self.assertNotIn("_evaluation_private", access)
        self.assertEqual(rows[0]["source_node_id"], 7)
        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(
            evaluator.call_args.kwargs,
            {
                "target_instance_id": 20,
                "include_evaluation_private_blockers": True,
            },
        )

    def test_physical_blocker_matching_is_globally_one_to_one(self) -> None:
        instances = np.empty((4, 5), dtype=np.int64)
        instances.reshape(-1)[:10] = 10
        instances.reshape(-1)[10:] = 20
        shared = np.zeros_like(instances, dtype=bool)
        shared.reshape(-1)[:8] = True
        shared.reshape(-1)[10:18] = True
        alternative = np.zeros_like(instances, dtype=bool)
        alternative.reshape(-1)[:4] = True
        graph = {
            "nodes": [
                {
                    "node_id": 7,
                    "node_type": "object",
                    "footprint_mask": encode_binary_mask_rle(shared),
                },
                {
                    "node_id": 8,
                    "node_type": "object",
                    "footprint_mask": encode_binary_mask_rle(alternative),
                },
            ]
        }
        rows = _oracle_rows(
            graph,
            {
                "physical_blocker_instance_ids": [10, 20],
                "blocker_candidate_counts": {"10": 1, "20": 2},
                "eligible_candidate_count": 2,
            },
            instances,
        )
        self.assertEqual([row["source_node_id"] for row in rows], [7, 8])
        self.assertEqual([row["score"] for row in rows], [1.0, 0.5])
        self.assertAlmostEqual(rows[0]["evaluation_match_iou"], 8.0 / 18.0)
        self.assertAlmostEqual(rows[1]["evaluation_match_iou"], 0.4)

    def test_failure_diagnostic_is_class_aware_runtime_only_and_stateful(self) -> None:
        first = np.zeros((12, 14), dtype=bool)
        first[2:6, 3:8] = True
        target = np.zeros_like(first)
        target[8:10, 9:12] = True
        instances = np.full(first.shape, -1, dtype=np.int64)
        instances[first] = 10
        instances[target] = 20
        semantics = np.full(first.shape, 14, dtype=np.int64)
        semantics[first] = 1
        semantics[target] = 2

        def graph(first_id: int, target_id: int) -> dict:
            query = "target-query"
            return {
                "step": 0,
                "nodes": [
                    {
                        "node_id": first_id,
                        "node_type": "object",
                        "class_id": 1,
                        "footprint_mask": encode_binary_mask_rle(first),
                    },
                    {
                        "node_id": target_id,
                        "node_type": "object",
                        "class_id": 2,
                        "footprint_mask": encode_binary_mask_rle(target),
                    },
                    {"node_id": query, "node_type": "target_query"},
                ],
                "edges": [
                    {
                        "edge_type": "candidate_of",
                        "source": target_id,
                        "target": query,
                    },
                    {
                        "edge_type": "blocks_access_to",
                        "source": first_id,
                        "target": target_id,
                        "score": 0.9,
                    },
                ],
            }

        config = {**DEFAULT_FAILURE_DIAGNOSTICS, "enabled": True}
        state: dict = {}
        diagnostic = _failure_diagnostic(
            graph=graph(7, 8),
            instance_map=instances,
            semantic_map=semantics,
            private={
                "physical_blocker_instance_ids": [10],
                "blocker_candidate_counts": {"10": 3},
                "eligible_candidate_count": 6,
            },
            target_private_id=20,
            selected_action={"kind": "push", "source_node_id": 7},
            config=config,
            state=state,
        )

        self.assertEqual(diagnostic["matched_object_count"], 2)
        self.assertEqual(diagnostic["target_node_id"], 8)
        self.assertTrue(diagnostic["target_candidate_edge_present"])
        self.assertEqual(diagnostic["edge_false_positive_count"], 0)
        self.assertEqual(diagnostic["edge_false_negative_count"], 0)
        self.assertIs(diagnostic["selected_source_is_physical_blocker"], True)
        self.assertEqual(
            diagnostic["physical_blocker_rows"],
            [{"source_node_id": 7, "score": 0.5, "evaluation_match_iou": 1.0}],
        )
        self.assertNotIn("evaluation_object_id", json.dumps(diagnostic))
        self.assertNotIn("instance_id", json.dumps(diagnostic))
        self.assertNotIn("20", json.dumps(diagnostic))

        changed = _failure_diagnostic(
            graph=graph(17, 18),
            instance_map=instances,
            semantic_map=semantics,
            private={
                "physical_blocker_instance_ids": [10],
                "blocker_candidate_counts": {"10": 3},
                "eligible_candidate_count": 6,
            },
            target_private_id=20,
            selected_action={"kind": "push", "source_node_id": 17},
            config=config,
            state=state,
        )
        self.assertEqual(changed["association_switch_count_current"], 2)
        self.assertEqual(changed["target_association_switch_count_cumulative"], 1)

    def test_failure_diagnostic_reports_target_merge_without_private_ids(self) -> None:
        left = np.zeros((8, 12), dtype=bool)
        left[2:5, 2:4] = True
        right = np.zeros_like(left)
        right[2:5, 4:6] = True
        merged = np.logical_or(left, right)
        instances = np.full(left.shape, -1, dtype=np.int64)
        instances[left] = 31
        instances[right] = 32
        semantics = np.full(left.shape, 14, dtype=np.int64)
        semantics[merged] = 3
        diagnostic = _failure_diagnostic(
            graph={
                "step": 0,
                "nodes": [
                    {
                        "node_id": "merged-track",
                        "node_type": "object",
                        "class_id": 3,
                        "footprint_mask": encode_binary_mask_rle(merged),
                    }
                ],
                "edges": [],
            },
            instance_map=instances,
            semantic_map=semantics,
            private={},
            target_private_id=31,
            selected_action={"kind": "observe"},
            config={**DEFAULT_FAILURE_DIAGNOSTICS, "enabled": True},
            state={},
        )
        self.assertEqual(diagnostic["node_merge_count"], 1)
        self.assertIs(diagnostic["target_merge"], True)
        self.assertFalse(diagnostic["simulator_ids_returned"])
        self.assertFalse(diagnostic["raw_maps_returned"])


if __name__ == "__main__":
    unittest.main()
