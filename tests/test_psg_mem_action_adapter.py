from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from shelf_gym.utils.psg_mem_action_adapter import (
    MemActionExecutionState,
    OfficialMemActionExecutor,
    PsgMemActionAdapter,
    build_frozen_mem_action_candidates,
    score_official_mem_push_candidates,
)


def _candidate():
    return {
        "candidate_id": "p1",
        "source_node_id": 4,
        "feasibility": 0.9,
        "action": {"kind": "push", "candidate_index": 1},
    }


class PsgMemActionAdapterTest(unittest.TestCase):
    def test_default_off_flag_must_be_a_real_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "enabled"):
            PsgMemActionAdapter(lambda **kwargs: {}, config={"enabled": "false"})

    def test_default_off_is_inert_and_leaves_official_caller_in_control(self) -> None:
        provider_calls = []
        execution_calls = []
        adapter = PsgMemActionAdapter(lambda **kwargs: provider_calls.append(kwargs))
        result = adapter.step(
            graph={"schema": "psg_mem_graph_v1"},
            push_candidates=[_candidate()],
            official_action=None,
            execute_mem_action=lambda action: execution_calls.append(action),
            arm_id="d",
            target_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(provider_calls, [])
        self.assertEqual(execution_calls, [])
        self.assertIs(result["delegates_to_official_caller"], True)
        self.assertIs(result["executes_action"], False)

    def test_enabled_treatment_executes_exactly_one_offered_action(self) -> None:
        candidate = _candidate()
        execution_calls = []

        def provider(**kwargs):
            self.assertEqual(
                kwargs["target_query"], {"class_id": 2, "coarse_region": "back"}
            )
            return {
                "selected_candidate_id": "p1",
                "selected_action": candidate["action"],
                "decision_source": "psg_mem",
                "executes_action": False,
            }

        adapter = PsgMemActionAdapter(provider, config={"enabled": True})
        result = adapter.step(
            graph={"schema": "psg_mem_graph_v1"},
            push_candidates=[candidate],
            official_action=None,
            execute_mem_action=lambda action: execution_calls.append(action)
            or {"ok": True},
            arm_id="d",
            target_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(execution_calls, [candidate["action"]])
        self.assertIs(result["executes_action"], True)
        self.assertIs(result["decision"]["executes_action"], True)

    def test_adapter_rejects_unoffered_or_privileged_decisions(self) -> None:
        adapter = PsgMemActionAdapter(
            lambda **kwargs: {
                "selected_candidate_id": "invented",
                "selected_action": {"kind": "push", "candidate_index": 99},
                "decision_source": "psg_mem",
                "executes_action": False,
            },
            config={"enabled": True},
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            adapter.step(
                graph={},
                push_candidates=[_candidate()],
                official_action=None,
                execute_mem_action=lambda action: None,
                arm_id="d",
                target_query={"class_id": 2, "coarse_region": "back"},
            )
        with self.assertRaisesRegex(ValueError, "privileged"):
            adapter.step(
                graph={},
                push_candidates=[_candidate()],
                official_action=None,
                execute_mem_action=lambda action: None,
                arm_id="d",
                target_query={"class_id": 2, "evaluation_object_id": 123},
            )

    def test_candidate_identity_is_type_stable(self) -> None:
        candidates = [_candidate(), {**_candidate(), "candidate_id": 1}]
        candidates[0]["candidate_id"] = "1"
        selected = candidates[1]
        adapter = PsgMemActionAdapter(
            lambda **kwargs: {
                "selected_candidate_id": 1,
                "selected_action": selected["action"],
                "decision_source": "psg_mem",
                "executes_action": False,
            },
            config={"enabled": True},
        )
        calls = []
        result = adapter.step(
            graph={},
            push_candidates=candidates,
            official_action=None,
            execute_mem_action=lambda action: calls.append(action),
            arm_id="d",
            target_query={"class_id": 2, "coarse_region": "back"},
        )
        self.assertEqual(calls, [selected["action"]])
        self.assertEqual(result["executed_candidate_id"], 1)

    def test_frozen_mem_candidates_encode_official_contact_coordinates(self) -> None:
        payload = {
            "paths": [np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])],
            "path_annotations": [["pushing"], ["pushing"]],
            "motion_parametrization": np.asarray(
                [[5, 180, 44, 10, 10, 170], [8, 175, 44, 20, 20, 165]],
                dtype=np.float32,
            ),
            "possible_previous_maps": np.zeros((2, 4, 3, 3), dtype=np.float32),
            "possible_semantic_maps": np.zeros((2, 5, 3, 3), dtype=np.float32),
        }
        candidates = build_frozen_mem_action_candidates(
            payload,
            viewpoint=9,
            viewpoint_information_gain=4.5,
            push_information_gains=[1.0, 2.0],
            include_noop=True,
        )
        self.assertEqual(candidates[0]["action"], {"kind": "observe", "viewpoint": 9})
        self.assertEqual(candidates[1]["contact_start_yx"], [15.0, 20.0])
        self.assertEqual(candidates[2]["contact_start_yx"], [18.0, 25.0])
        self.assertEqual(candidates[1]["contact_end_yx"], [54.0, 190.0])
        self.assertEqual(candidates[1]["push_direction_xy"], [170.0, 39.0])
        self.assertEqual(candidates[1]["action"]["push_direction_xy"], [170.0, 39.0])
        self.assertEqual(
            candidates[1]["contact_coordinate_frame"]["crop_row_offset"], 10
        )
        self.assertEqual(candidates[2]["action"]["candidate_index"], 1)
        self.assertEqual(len(candidates[2]["action"]["candidate_fingerprint"]), 64)
        self.assertEqual(candidates[3]["action"]["kind"], "noop")

    def test_official_executor_runs_exact_selected_push_then_forced_view(self) -> None:
        predicted_maps = np.stack([np.full((4, 2, 2), 1.0), np.full((4, 2, 2), 2.0)])
        semantic_maps = np.stack([np.full((5, 2, 2), 3.0), np.full((5, 2, 2), 4.0)])
        payload = {
            "paths": [np.asarray([[1.0]]), np.asarray([[2.0]])],
            "path_annotations": [["pushing"], ["pushing", "pushing"]],
            "motion_parametrization": np.asarray(
                [[5, 180, 44, 10, 10, 170], [8, 175, 44, 20, 20, 165]],
                dtype=np.float32,
            ),
            "possible_previous_maps": predicted_maps,
            "possible_semantic_maps": semantic_maps,
        }
        candidates = build_frozen_mem_action_candidates(
            payload,
            viewpoint=9,
            viewpoint_information_gain=4.5,
            push_information_gains=[1.0, 2.0],
        )
        calls = []

        class FakeObjects:
            def check_all_object_drop(self, object_ids):
                calls.append(("drop", list(object_ids)))
                return False

        class FakeMem:
            current_obj_ids = [101, 102]
            obj = FakeObjects()

            def execute_observation(self, views, viewpoint, previous, semantic):
                calls.append(("observe", viewpoint, list(views)))
                views.append(viewpoint)
                return previous + 10, semantic + 20

        state = MemActionExecutionState(
            previous_map=np.zeros((1, 4, 2, 2), dtype=np.float32),
            previous_semantic_map=np.zeros((1, 5, 2, 2), dtype=np.float32),
            previous_views=[3],
        )

        def execute_push(mem, path, *, path_annotations):
            calls.append(("push", path.tolist(), list(path_annotations)))
            return (0, "ok")

        executor = OfficialMemActionExecutor(
            FakeMem(), state=state, execute_push=execute_push
        )
        executor.set_push_candidates(payload)
        pushed = executor.execute(candidates[2]["action"])
        self.assertEqual(pushed["push_return_code"], 0)
        self.assertIs(state.fresh_push, True)
        self.assertEqual(state.previous_views, [])
        np.testing.assert_array_equal(state.previous_map, predicted_maps[1][None])
        observed = executor.execute(candidates[0]["action"])
        self.assertEqual(observed["action_kind"], "observe")
        self.assertIs(state.fresh_push, False)
        self.assertEqual(state.previous_views, [9])
        self.assertEqual([call[0] for call in calls], ["push", "drop", "observe"])

    def test_official_executor_rejects_stale_candidate_token(self) -> None:
        payload = {
            "paths": [np.asarray([[1.0]])],
            "path_annotations": [["pushing"]],
            "motion_parametrization": np.asarray(
                [[5, 180, 44, 10, 10, 170]], dtype=np.float32
            ),
            "possible_previous_maps": np.zeros((1, 4, 2, 2), dtype=np.float32),
            "possible_semantic_maps": np.zeros((1, 5, 2, 2), dtype=np.float32),
        }
        candidate = build_frozen_mem_action_candidates(
            payload,
            viewpoint=9,
            viewpoint_information_gain=4.5,
            push_information_gains=[1.0],
        )[1]
        state = MemActionExecutionState(
            previous_map=np.zeros((1, 4, 2, 2)),
            previous_semantic_map=np.zeros((1, 5, 2, 2)),
        )
        executor = OfficialMemActionExecutor(
            object(), state=state, execute_push=lambda *args, **kwargs: (0,)
        )
        executor.set_push_candidates(payload)
        stale = dict(candidate["action"])
        stale["candidate_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale|fingerprint"):
            executor.execute(stale)

    def test_official_push_scorer_captures_existing_evaluator_without_rescoring(
        self,
    ) -> None:
        push_calculator = object()
        view_calculator = object()
        calls = []

        def official_get_igs(previous_map, calculator, **kwargs):
            calls.append(calculator)
            if calculator is push_calculator:
                index = sum(value is push_calculator for value in calls) - 1
                return np.asarray([index + 0.5, index + 1.5]), None
            return np.asarray([4.0, 3.0]), None

        pipeline = SimpleNamespace(get_igs_for_map=official_get_igs)

        class FakeMem:
            push_ig_calc = push_calculator
            ig_calc = view_calculator

            def eval_push_igs(self, push_data, semantic_map, **kwargs):
                for possible in push_data["possible_previous_maps"]:
                    pipeline.get_igs_for_map(possible, self.push_ig_calc, skip=5)
                pipeline.get_igs_for_map(
                    push_data["possible_previous_maps"][1], self.ig_calc, skip=1
                )
                return 0, 1, 4.0

        original = pipeline.get_igs_for_map
        result = score_official_mem_push_candidates(
            FakeMem(),
            pipeline,
            {
                "paths": [object(), object()],
                "possible_previous_maps": [np.zeros(1), np.ones(1)],
            },
            previous_semantic_map=np.zeros(1),
            use_delta_H=True,
            skip=5,
        )
        self.assertIs(pipeline.get_igs_for_map, original)
        self.assertEqual(result["official_candidate_index"], 1)
        self.assertEqual(result["candidate_information_gains"], [1.5, 2.5])
        self.assertEqual(len(calls), 3)

    def test_official_executor_preserves_planner_noop_without_physical_call(
        self,
    ) -> None:
        state = MemActionExecutionState(
            previous_map=np.zeros((1, 4, 2, 2)),
            previous_semantic_map=np.zeros((1, 5, 2, 2)),
        )
        executor = OfficialMemActionExecutor(
            object(),
            state=state,
            execute_push=lambda *args, **kwargs: self.fail("push must stay inert"),
        )
        result = executor.execute(
            {"kind": "noop", "reason": "push_not_better_than_view_horizon"}
        )
        self.assertIs(result["executes_action"], False)
        self.assertIs(result["action_budget_consumed"], True)
        self.assertEqual(state.action_count, 1)


if __name__ == "__main__":
    unittest.main()
