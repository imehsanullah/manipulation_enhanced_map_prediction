from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from shelf_gym.utils.psg_mem_action_adapter import (
    MemActionExecutionState,
    OfficialMemActionExecutor,
)
from shelf_gym.utils.psg_mem_live_bridge import OfficialMemStepBridge


def _push_payload():
    return {
        "paths": [[[1.0], [2.0]]],
        "path_annotations": [["pushing", "pushing"]],
        "motion_parametrization": np.asarray(
            [[5, 180, 44, 10, 10, 170]], dtype=np.float32
        ),
        "possible_previous_maps": np.full((1, 4, 2, 2), 3.0),
        "possible_semantic_maps": np.full((1, 5, 2, 2), 4.0),
    }


class _Objects:
    def check_all_object_drop(self, object_ids):
        return False


class _Mem:
    ig_calc = object()
    current_obj_ids = [1]
    obj = _Objects()
    stopping_criterion = 0.99

    def __init__(self):
        self.push_requests = []

    def get_possible_maps_push(self, previous, semantic, **kwargs):
        self.push_requests.append(kwargs)
        return _push_payload()

    def execute_observation(self, views, viewpoint, previous, semantic):
        views.append(viewpoint)
        return previous + 1, semantic + 1

    def get_semantic_certainty(self, semantic):
        return semantic

    def get_certainly_mapped_fraction(self, certainty, cutoff):
        return 0.25

    prob_cutoff = 0.85


class OfficialMemStepBridgeTest(unittest.TestCase):
    def _bridge(
        self,
        *,
        official_push_ig=4.0,
        enabled=True,
        action_budget=2,
        first_push_step=0,
        treatment_first_push_step=0,
    ):
        mem = _Mem()
        state = MemActionExecutionState(
            previous_map=np.zeros((1, 4, 2, 2), dtype=np.float32),
            previous_semantic_map=np.ones((1, 5, 2, 2), dtype=np.float32),
        )
        executor = OfficialMemActionExecutor(
            mem,
            state=state,
            execute_push=lambda mem, path, **kwargs: (0, None),
        )
        pipeline = SimpleNamespace(
            get_igs_for_map=lambda previous, calculator, **kwargs: (
                np.asarray([1.0, 5.0]),
                None,
            ),
            get_subsequent_igs_for_map=lambda previous, views, calculator: np.asarray(
                [2.0, 1.0]
            ),
        )
        graph_calls = []

        def graph_provider(**kwargs):
            graph_calls.append(
                copy.deepcopy(
                    {
                        key: value
                        for key, value in kwargs.items()
                        if key
                        not in {
                            "occupancy_distribution",
                            "semantic_concentration",
                            "push_data",
                        }
                    }
                )
            )
            return {
                "schema": "psg_mem_graph_v1",
                "episode_id": kwargs["episode_id"],
                "step": kwargs["step"],
            }

        def bind_candidates(graph, candidates):
            result = copy.deepcopy(candidates)
            for candidate in result:
                if candidate["candidate_type"] == "push":
                    candidate["source_node_id"] = 7
            return result

        bridge = OfficialMemStepBridge(
            mem,
            state=state,
            action_executor=executor,
            pipeline_module=pipeline,
            graph_provider=graph_provider,
            candidate_binder=bind_candidates,
            push_score_provider=lambda *args, **kwargs: {
                "official_viewpoint_after_push": 0,
                "official_candidate_index": 0,
                "official_push_information_gain": official_push_ig,
                "candidate_information_gains": [3.0],
            },
            episode_id="episode-1",
            target_query={"class_id": 2, "coarse_region": "back"},
            config={
                "enabled": enabled,
                "action_budget": action_budget,
                "first_push_step": first_push_step,
                "treatment_first_push_step": treatment_first_push_step,
            },
        )
        return bridge, mem, state, graph_calls

    def test_default_off_bridge_is_inert(self) -> None:
        bridge, mem, state, graph_calls = self._bridge(enabled=False)
        result = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=0,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertIs(result["enabled"], False)
        self.assertIs(result["delegates_to_official_mem_run"], True)
        self.assertEqual(mem.push_requests, [])
        self.assertEqual(graph_calls, [])
        self.assertEqual(state.action_count, 0)

    def test_live_bridge_preserves_official_noop_but_offers_same_push(self) -> None:
        bridge, mem, state, graph_calls = self._bridge(official_push_ig=4.0)
        payload = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=0,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(payload["official_action"]["action"]["kind"], "noop")
        self.assertEqual(
            [row["candidate_type"] for row in payload["action_candidates"]],
            ["view", "push", "noop"],
        )
        self.assertEqual(payload["action_candidates"][1]["source_node_id"], 7)
        self.assertEqual(mem.push_requests, [{"planner_camera_index": 1}])
        self.assertEqual(len(graph_calls), 1)
        executed = bridge.execute_mem_action(payload["official_action"]["action"])
        self.assertIs(executed["executes_action"], False)
        self.assertEqual(state.action_count, 1)

    def test_treatment_push_uses_official_executor_and_forces_next_view(self) -> None:
        bridge, _mem, state, _graph_calls = self._bridge(official_push_ig=8.0)
        payload = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=0,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        push = next(
            row
            for row in payload["action_candidates"]
            if row["candidate_type"] == "push"
        )
        result = bridge.execute_mem_action(push["action"])
        self.assertEqual(result["action_kind"], "push")
        self.assertIs(state.fresh_push, True)
        next_payload = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=1,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(
            [row["candidate_type"] for row in next_payload["action_candidates"]],
            ["view"],
        )
        self.assertEqual(next_payload["official_action"]["action"]["kind"], "observe")

    def test_treatment_gets_early_push_without_changing_official_view(self) -> None:
        bridge, mem, _state, _graph_calls = self._bridge(
            official_push_ig=99.0,
            action_budget=8,
            first_push_step=3,
            treatment_first_push_step=1,
        )
        warmup = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=0,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(
            [row["candidate_type"] for row in warmup["action_candidates"]],
            ["view"],
        )
        bridge.execute_mem_action(warmup["official_action"]["action"])
        early = bridge.step_provider(
            arm_id="d",
            episode_id="episode-1",
            step=1,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(early["official_action"]["action"]["kind"], "observe")
        self.assertEqual(
            [row["candidate_type"] for row in early["action_candidates"]],
            ["view", "push"],
        )
        self.assertIs(
            early["bridge_diagnostics"]["treatment_early_push_opportunity"], True
        )
        self.assertEqual(len(mem.push_requests), 1)

    def test_official_arm_retains_no_early_push_candidate_generation(self) -> None:
        bridge, mem, _state, _graph_calls = self._bridge(
            official_push_ig=99.0,
            action_budget=8,
            first_push_step=3,
            treatment_first_push_step=1,
        )
        payload = bridge.step_provider(
            arm_id="a",
            episode_id="episode-1",
            step=0,
            planner_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(payload["official_action"]["action"]["kind"], "observe")
        self.assertEqual(
            [row["candidate_type"] for row in payload["action_candidates"]],
            ["view"],
        )
        self.assertEqual(mem.push_requests, [])


if __name__ == "__main__":
    unittest.main()
