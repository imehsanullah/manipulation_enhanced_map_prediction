import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "shelf_gym"
    / "scripts"
    / "run_cnabu_mem_baseline_smoke.py"
)
_SPEC = importlib.util.spec_from_file_location("run_cnabu_mem_baseline_smoke", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
mean_iou = _MODULE.mean_iou
numpy_choice_fallback = _MODULE._numpy_choice_fallback
configure_cupy_choice = _MODULE._configure_cupy_choice
to_numpy_cpu = _MODULE._to_numpy_cpu
summarise_episode = _MODULE._summarise_episode


def test_mean_iou_uses_only_classes_present_in_prediction_or_target():
    result = mean_iou(
        np.asarray([[0, 1], [1, 1]]),
        np.asarray([[0, 0], [1, 1]]),
        num_classes=3,
    )

    assert result["present_class_count"] == 2
    assert result["per_class"][0]["iou"] == pytest.approx(0.5)
    assert result["per_class"][1]["iou"] == pytest.approx(2.0 / 3.0)
    assert result["per_class"][2]["iou"] is None
    assert result["macro_iou"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


def test_mean_iou_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        mean_iou(np.zeros((2, 2)), np.zeros((2, 3)), num_classes=2)


def test_numpy_choice_fallback_is_seeded_uniform_without_replacement():
    class FakeCupy:
        asarray = staticmethod(np.asarray)
        asnumpy = staticmethod(np.asarray)

    first = numpy_choice_fallback(7, FakeCupy)(10, size=4, replace=False)
    second = numpy_choice_fallback(7, FakeCupy)(10, size=4, replace=False)

    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 4


def test_controlled_choice_forces_seeded_probability_capable_numpy_backend():
    def original_choice(*_args, **_kwargs):
        raise AssertionError("native choice must not execute")

    class FakeRandom:
        choice = staticmethod(original_choice)

    class FakeCupy:
        random = FakeRandom()
        asarray = staticmethod(np.asarray)
        asnumpy = staticmethod(np.asarray)

    with patch.dict(sys.modules, {"cupy": FakeCupy}):
        configured = configure_cupy_choice(11, force_numpy=True)
        first = FakeCupy.random.choice(
            4, size=2, replace=False, p=np.asarray([0.1, 0.2, 0.3, 0.4])
        )
        FakeCupy.random.choice = configured["original"]

    assert configured["backend"] == "numpy_random_state_choice"
    assert configured["supports_probability_vector"]
    assert first.shape == (2,)


def test_to_numpy_cpu_uses_explicit_get_for_cupy_like_arrays():
    class FakeCupyArray:
        def get(self):
            return np.asarray([1, 2, 3])

        def __array__(self):
            raise AssertionError("implicit conversion must not be used")

    assert np.array_equal(to_numpy_cpu(FakeCupyArray()), np.asarray([1, 2, 3]))


def test_summarise_episode_uses_store_results_aligned_gt_without_recropping():
    occupancy = np.zeros((84, 158, 59), dtype=np.float32)
    occupancy_gt = np.zeros_like(occupancy)
    semantics = np.zeros((84, 158, 3), dtype=np.float32)
    semantics[..., 0] = 1.0
    semantics_gt = np.zeros((84, 158), dtype=np.int64)

    result = summarise_episode(
        {
            "occupancy_map": [occupancy],
            "semantic_map": [semantics],
            "occupancy_gt": [occupancy_gt],
            "semantic_gt": [semantics_gt],
            "step_time": [1.0],
            "vpp_time": [0.5],
            "push_time": [0.0],
            "pushes": [0],
        },
        n_classes=3,
        step_action_trace=[None],
    )

    assert result["steps_completed"] == 1
    assert result["final_occupancy_macro_iou"] == pytest.approx(1.0)
    assert result["final_semantic_macro_iou"] == pytest.approx(1.0)
    assert result["step_action_codes"] == [None]
    assert result["num_planner_noop_steps"] == 1


def test_summarise_episode_reports_push_safety_and_selection_provenance():
    occupancy = np.zeros((2, 2, 1), dtype=np.float32)
    semantics = np.ones((2, 2, 1), dtype=np.float32)
    result = summarise_episode(
        {
            "occupancy_map": [occupancy],
            "semantic_map": [semantics],
            "occupancy_gt": [occupancy.copy()],
            "semantic_gt": [np.zeros((2, 2), dtype=np.int64)],
            "step_time": [1.0],
            "vpp_time": [0.0],
            "push_time": [1.0],
            "pushes": [2],
        },
        n_classes=1,
        step_action_trace=[2],
        step_safety_trace=[
            {
                "attempted_push": True,
                "push_return_code": 0,
                "tilted_object_failure": False,
                "object_drop": False,
                "accepted_without_tilt_or_drop": True,
                "candidate_selected_but_not_executed": False,
                "selected_candidate": {
                    "candidate_index": 1,
                    "sampling_priority_provenance": "deterministic",
                },
            }
        ],
        mapping_completion_trace=[0.91],
        mapping_completion_threshold=0.90,
    )

    assert result["push_safety"]["attempted_push_count"] == 1
    assert result["push_safety"]["accepted_without_tilt_or_drop_count"] == 1
    assert result["push_safety"]["failed_push_count"] == 0
    assert result["action_count_to_mapping_threshold"] == 1
    assert result["mapping_threshold_reached"]
    assert result["steps"][0]["safety"]["selected_candidate"][
        "sampling_priority_provenance"
    ] == "deterministic"
