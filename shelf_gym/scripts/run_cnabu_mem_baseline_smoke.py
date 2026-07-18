"""Run one compact, reproducible original-MEM baseline episode.

This wrapper intentionally does not alter candidate generation, push scoring,
VIG scoring, action selection, or execution.  It adds phase timers around the
existing implementation and reduces the returned dense maps to compact metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from shelf_gym.utils.cnabu_mem_experiment_control import (
    apply_initial_state_snapshot,
    candidate_set_fingerprint,
    configure_controlled_mem,
    configure_deterministic_process_environment,
    initial_observation_hashes,
    load_initial_state_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_PATH = Path(__file__).resolve().parent / "model" / "camera_matrices.npz"
DEFAULT_MAP_CHECKPOINT = Path(__file__).resolve().parent / "model" / "model-5dburcae:v4.ckpt"
DEFAULT_PUSH_CHECKPOINT = Path(__file__).resolve().parent / "model" / "push_predictor_new.ckpt"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def mean_iou(
    prediction: Any,
    target: Any,
    *,
    num_classes: int,
) -> Dict[str, Any]:
    """Return macro IoU over classes present in prediction or target."""

    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.shape != expected.shape:
        raise ValueError(
            "prediction and target must have the same shape, got {} and {}".format(
                predicted.shape, expected.shape
            )
        )
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")

    per_class = []
    present_values = []
    for class_id in range(int(num_classes)):
        predicted_mask = predicted == class_id
        target_mask = expected == class_id
        intersection = int(np.logical_and(predicted_mask, target_mask).sum())
        union = int(np.logical_or(predicted_mask, target_mask).sum())
        value = None if union == 0 else float(intersection / union)
        per_class.append(
            {
                "class_id": int(class_id),
                "intersection": intersection,
                "union": union,
                "iou": value,
            }
        )
        if value is not None:
            present_values.append(value)
    return {
        "macro_iou": float(np.mean(present_values)) if present_values else None,
        "present_class_count": int(len(present_values)),
        "per_class": per_class,
    }


def _to_numpy_cpu(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()

    return {
        "path": str(repo),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import cupy as cp

        cp.random.seed(seed)
    except (ImportError, RuntimeError):
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numpy_choice_fallback(seed: int, cupy_module: Any) -> Any:
    """Return a CuPy-compatible uniform choice backed by an isolated NumPy RNG."""

    generator = np.random.RandomState(int(seed))

    def choice(a: Any, size: Any = None, replace: bool = True, p: Any = None) -> Any:
        if np.isscalar(a):
            population = int(a)
        else:
            population = np.asarray(cupy_module.asnumpy(a))
        probabilities = None if p is None else np.asarray(cupy_module.asnumpy(p))
        return cupy_module.asarray(
            generator.choice(population, size=size, replace=replace, p=probabilities)
        )

    return choice


def _configure_cupy_choice(
    seed: int,
    *,
    force_numpy: bool = False,
) -> Dict[str, Any]:
    """Use cuRAND when available, otherwise preserve uniform sampling on CPU."""

    import cupy as cp

    original = cp.random.choice
    if force_numpy:
        cp.random.choice = _numpy_choice_fallback(int(seed), cp)
        return {
            "backend": "numpy_random_state_choice",
            "fallback": True,
            "fallback_reason": "forced_controlled_cross_arm_weighted_choice_contract",
            "supports_probability_vector": True,
            "original": original,
        }
    try:
        cp.random.seed(int(seed))
        probe = cp.random.choice(8, 2, replace=False)
        cp.cuda.get_current_stream().synchronize()
        _ = probe.get()
        return {
            "backend": "cupy_curand",
            "fallback": False,
            "fallback_reason": None,
            "supports_probability_vector": False,
            "original": original,
        }
    except Exception as exc:  # CUDA/cuRAND compatibility errors vary by CuPy build.
        cp.random.choice = _numpy_choice_fallback(int(seed), cp)
        return {
            "backend": "numpy_random_state_uniform_choice",
            "fallback": True,
            "fallback_reason": "{}: {}".format(type(exc).__name__, exc),
            "supports_probability_vector": True,
            "original": original,
        }


def _synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PhaseRecorder:
    def __init__(self) -> None:
        self._records: MutableMapping[str, Dict[str, Any]] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        _synchronize_cuda()
        started = time.perf_counter()
        try:
            yield
        finally:
            _synchronize_cuda()
            elapsed = float(time.perf_counter() - started)
            record = self._records.setdefault(name, {"calls": 0, "wall_seconds": 0.0})
            record["calls"] = int(record["calls"]) + 1
            record["wall_seconds"] = float(record["wall_seconds"]) + elapsed

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "calls": int(record["calls"]),
                "wall_seconds": float(record["wall_seconds"]),
            }
            for name, record in sorted(self._records.items())
        }


def _wrap_callable(owner: Any, attribute: str, recorder: PhaseRecorder, phase: str) -> Any:
    original = getattr(owner, attribute)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with recorder.measure(phase):
            return original(*args, **kwargs)

    setattr(owner, attribute, wrapped)
    return original


def _restore_callable(owner: Any, attribute: str, original: Any) -> None:
    setattr(owner, attribute, original)


def _summarise_episode(
    output: Mapping[str, Sequence[Any]],
    *,
    n_classes: int,
    step_action_trace: Sequence[int | None] | None = None,
    belief_trace: Sequence[Mapping[str, float]] | None = None,
    step_safety_trace: Sequence[Mapping[str, Any]] | None = None,
    mapping_completion_trace: Sequence[float | None] | None = None,
    mapping_completion_threshold: float | None = None,
) -> Dict[str, Any]:
    action_trace = (
        [None for _ in output["occupancy_map"]]
        if step_action_trace is None
        else list(step_action_trace)
    )
    if len(action_trace) != len(output["occupancy_map"]):
        raise ValueError("step action trace must align with stored map steps")
    if belief_trace is not None and len(belief_trace) != len(output["occupancy_map"]):
        raise ValueError("belief trace must align with stored map steps")
    if step_safety_trace is not None and len(step_safety_trace) != len(
        output["occupancy_map"]
    ):
        raise ValueError("step safety trace must align with stored map steps")
    if mapping_completion_trace is not None and len(mapping_completion_trace) != len(
        output["occupancy_map"]
    ):
        raise ValueError("mapping completion trace must align with stored map steps")
    if (mapping_completion_trace is None) != (mapping_completion_threshold is None):
        raise ValueError("mapping completion trace and threshold must be paired")
    step_records = []
    for step_index, (occupancy, semantics, occupancy_gt, semantics_gt) in enumerate(
        zip(
            output["occupancy_map"],
            output["semantic_map"],
            output["occupancy_gt"],
            output["semantic_gt"],
        )
    ):
        occupancy_array = _to_numpy_cpu(occupancy)
        semantics_array = _to_numpy_cpu(semantics)
        # ``ManipulationEnhancedMapping.store_results`` has already aligned both
        # predictions and GT to the evaluation crop.  Cropping the returned GT a
        # second time silently changes the benchmark support.
        occupancy_gt_array = _to_numpy_cpu(occupancy_gt)
        semantics_gt_array = _to_numpy_cpu(semantics_gt)
        if occupancy_array.shape != occupancy_gt_array.shape:
            raise ValueError(
                "stored occupancy and cropped GT shapes differ: {} versus {}".format(
                    occupancy_array.shape, occupancy_gt_array.shape
                )
            )
        if semantics_array.shape[:2] != semantics_gt_array.shape:
            raise ValueError(
                "stored semantics and cropped GT shapes differ: {} versus {}".format(
                    semantics_array.shape[:2], semantics_gt_array.shape
                )
            )
        occupancy_metrics = mean_iou(
            occupancy_array >= 0.5,
            occupancy_gt_array >= 0.5,
            num_classes=2,
        )
        semantic_metrics = mean_iou(
            semantics_array.argmax(axis=-1),
            semantics_gt_array,
            num_classes=n_classes,
        )
        target_occupied = occupancy_gt_array >= 0.5
        occupied_denominator = int(target_occupied.sum())
        occupied_recall = (
            float(np.logical_and(occupancy_array >= 0.5, target_occupied).sum())
            / occupied_denominator
            if occupied_denominator
            else None
        )
        belief_record = None if belief_trace is None else belief_trace[step_index]
        safety_record = (
            None
            if step_safety_trace is None
            else dict(step_safety_trace[step_index])
        )
        step_records.append(
            {
                "step_index": int(step_index),
                "occupancy_macro_iou": occupancy_metrics["macro_iou"],
                "semantic_macro_iou": semantic_metrics["macro_iou"],
                "semantic_confident_fraction_0p85": float(
                    (semantics_array.max(axis=-1) >= 0.85).mean()
                ),
                "gt_occupied_support_recall": occupied_recall,
                "occupancy_epistemic_mean": (
                    None
                    if belief_record is None
                    else float(belief_record["occupancy_epistemic_mean"])
                ),
                "semantic_vacuity_mean": (
                    None
                    if belief_record is None
                    else float(belief_record["semantic_vacuity_mean"])
                ),
                "occupancy_iou": occupancy_metrics,
                "semantic_iou": semantic_metrics,
                "step_seconds": float(output["step_time"][step_index]),
                "viewpoint_planning_seconds": float(output["vpp_time"][step_index]),
                "push_pipeline_seconds": float(output["push_time"][step_index]),
                "action_code": (
                    int(action_trace[step_index])
                    if action_trace[step_index] is not None
                    else None
                ),
                "safety": safety_record,
                "certainly_mapped_fraction": (
                    None
                    if mapping_completion_trace is None
                    or mapping_completion_trace[step_index] is None
                    else float(mapping_completion_trace[step_index])
                ),
            }
        )

    def auc(key: str) -> float | None:
        values = [record[key] for record in step_records]
        if not values or any(value is None for value in values):
            return None
        if len(values) == 1:
            return float(values[0])
        return float(np.trapz(np.asarray(values, dtype=np.float64), dx=1.0))

    safety_records = (
        [] if step_safety_trace is None else [dict(value) for value in step_safety_trace]
    )
    threshold_action_count = None
    if mapping_completion_trace is not None:
        for index, value in enumerate(mapping_completion_trace):
            if value is not None and float(value) >= float(mapping_completion_threshold):
                threshold_action_count = int(index + 1)
                break
    return {
        "steps": step_records,
        "steps_completed": int(len(step_records)),
        "step_action_codes": [
            int(value) if value is not None else None for value in action_trace
        ],
        "legacy_unaligned_push_action_codes": [
            int(value) for value in output["pushes"]
        ],
        "num_executed_pushes": int(sum(value == 2 for value in action_trace)),
        "num_planner_noop_steps": int(sum(value is None for value in action_trace)),
        "mapping_completion_threshold": (
            None
            if mapping_completion_threshold is None
            else float(mapping_completion_threshold)
        ),
        "action_count_to_mapping_threshold": threshold_action_count,
        "mapping_threshold_reached": bool(threshold_action_count is not None),
        "push_safety": {
            "instrumented": step_safety_trace is not None,
            "attempted_push_count": int(
                sum(bool(value.get("attempted_push", False)) for value in safety_records)
            ),
            "accepted_without_tilt_or_drop_count": int(
                sum(
                    bool(value.get("accepted_without_tilt_or_drop", False))
                    for value in safety_records
                )
            ),
            "failed_push_count": int(
                sum(
                    bool(value.get("attempted_push", False))
                    and not bool(value.get("accepted_without_tilt_or_drop", False))
                    for value in safety_records
                )
            ),
            "tilted_object_failure_count": int(
                sum(
                    bool(value.get("tilted_object_failure", False))
                    for value in safety_records
                )
            ),
            "object_drop_count": int(
                sum(bool(value.get("object_drop", False)) for value in safety_records)
            ),
            "candidate_selected_but_not_executed_count": int(
                sum(
                    bool(value.get("candidate_selected_but_not_executed", False))
                    for value in safety_records
                )
            ),
            "runtime_fixed_environment_collision_observed": None,
            "runtime_robot_collision_observed": None,
            "collision_observation_limitation": (
                "legacy execution exposes tilted-object and post-action object-drop "
                "signals, not separate runtime robot/fixed-environment contacts"
            ),
        },
        "occupancy_macro_iou_auc": auc("occupancy_macro_iou"),
        "semantic_macro_iou_auc": auc("semantic_macro_iou"),
        "semantic_confident_fraction_auc": auc("semantic_confident_fraction_0p85"),
        "gt_occupied_support_recall_auc": auc("gt_occupied_support_recall"),
        "occupancy_epistemic_mean_auc": auc("occupancy_epistemic_mean"),
        "semantic_vacuity_mean_auc": auc("semantic_vacuity_mean"),
        "final_occupancy_macro_iou": (
            step_records[-1]["occupancy_macro_iou"] if step_records else None
        ),
        "final_semantic_macro_iou": (
            step_records[-1]["semantic_macro_iou"] if step_records else None
        ),
        "final_semantic_confident_fraction_0p85": (
            step_records[-1]["semantic_confident_fraction_0p85"] if step_records else None
        ),
        "final_gt_occupied_support_recall": (
            step_records[-1]["gt_occupied_support_recall"] if step_records else None
        ),
        "occupancy_epistemic_reduction": (
            None
            if not step_records
            or step_records[0]["occupancy_epistemic_mean"] is None
            else float(
                step_records[0]["occupancy_epistemic_mean"]
                - step_records[-1]["occupancy_epistemic_mean"]
            )
        ),
        "semantic_vacuity_reduction": (
            None
            if not step_records or step_records[0]["semantic_vacuity_mean"] is None
            else float(
                step_records[0]["semantic_vacuity_mean"]
                - step_records[-1]["semantic_vacuity_mean"]
            )
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predefined-scene", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--action-budget", type=int, default=5)
    parser.add_argument("--max-sampled-pushes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-state-snapshot",
        type=Path,
        default=None,
        help="Opt into the frozen paired-run RNG/physics/input control policy.",
    )
    parser.add_argument("--disable-push", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    scene_path = args.predefined_scene.resolve()
    output_path = args.output_json.resolve()
    initial_state_path = (
        None
        if args.initial_state_snapshot is None
        else args.initial_state_snapshot.resolve()
    )
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if initial_state_path is not None and not initial_state_path.is_file():
        raise FileNotFoundError(initial_state_path)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    if int(args.action_budget) <= 0:
        raise ValueError("--action-budget must be positive")
    if int(args.max_sampled_pushes) <= 0:
        raise ValueError("--max-sampled-pushes must be positive")

    process_environment = (
        configure_deterministic_process_environment()
        if initial_state_path is not None
        else None
    )
    initial_state_snapshot = (
        load_initial_state_snapshot(initial_state_path)
        if initial_state_path is not None
        else None
    )
    _seed_everything(int(args.seed))
    import torch
    import shelf_gym.scripts.run_cnabu_pipeline as pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("the original MEM checkpoints require CUDA")
    torch.cuda.reset_peak_memory_stats()
    recorder = PhaseRecorder()
    started = time.perf_counter()
    init_started = time.perf_counter()
    mem = pipeline.ManipulationEnhancedMapping(
        render=bool(args.render),
        show_vis=False,
        use_uncertainty_informed_sampling=False,
    )
    initialization_seconds = float(time.perf_counter() - init_started)
    mem.action_budget = int(args.action_budget)
    mem.max_sampled_pushes = int(args.max_sampled_pushes)
    experiment_control = (
        configure_controlled_mem(mem, seed=int(args.seed))
        if initial_state_snapshot is not None
        else None
    )
    cupy_choice = _configure_cupy_choice(
        int(args.seed),
        force_numpy=initial_state_snapshot is not None,
    )

    originals = []
    hook_handles = []
    step_action_trace: list[int | None] = []
    belief_trace: list[Dict[str, float]] = []
    candidate_trace: list[Dict[str, Any]] = []
    push_decisions: list[Dict[str, Any]] = []
    push_execution_results: list[Dict[str, Any]] = []
    step_safety_trace: list[Dict[str, Any]] = []
    mapping_completion_calls: list[float] = []
    mapping_completion_trace: list[float | None] = []
    initial_state_application: Dict[str, Any] | None = None
    initial_input_hashes: Dict[str, Any] | None = None
    try:
        if initial_state_snapshot is not None:
            original_restore_shelf_state = mem.restore_shelf_state

            def controlled_restore_shelf_state(*restore_args: Any, **restore_kwargs: Any) -> Any:
                nonlocal initial_state_application
                result = original_restore_shelf_state(*restore_args, **restore_kwargs)
                initial_state_application = apply_initial_state_snapshot(
                    mem,
                    initial_state_snapshot,
                    scene_path=scene_path,
                )
                return result

            mem.restore_shelf_state = controlled_restore_shelf_state
            originals.append(
                (mem, "restore_shelf_state", original_restore_shelf_state)
            )

        for owner, attribute, phase in (
            (mem.ps, "get_samples", "push_candidate_generation"),
            (mem.ig_calc, "get_all_igs", "view_information_gain"),
            (mem.ig_calc, "get_all_subsequent_igs", "view_second_horizon_information_gain"),
            (mem.push_ig_calc, "get_all_igs", "push_information_gain"),
            (mem, "execute_observation", "observation_cnabu_update"),
        ):
            originals.append(
                (owner, attribute, _wrap_callable(owner, attribute, recorder, phase))
            )
        original_execute_push = pipeline.execute_push

        def traced_execute_push(*push_args: Any, **push_kwargs: Any) -> Any:
            with recorder.measure("push_execution"):
                result = original_execute_push(*push_args, **push_kwargs)
            code = int(result[0])
            push_execution_results.append(
                {
                    "push_return_code": code,
                    "tilted_object_failure": bool(code != 0),
                }
            )
            return result

        pipeline.execute_push = traced_execute_push
        originals.append((pipeline, "execute_push", original_execute_push))

        original_get_certainly_mapped_fraction = mem.get_certainly_mapped_fraction

        def traced_get_certainly_mapped_fraction(*mapped_args: Any, **mapped_kwargs: Any) -> Any:
            value = original_get_certainly_mapped_fraction(
                *mapped_args, **mapped_kwargs
            )
            mapping_completion_calls.append(float(value))
            return value

        mem.get_certainly_mapped_fraction = traced_get_certainly_mapped_fraction
        originals.append(
            (
                mem,
                "get_certainly_mapped_fraction",
                original_get_certainly_mapped_fraction,
            )
        )

        original_get_possible_maps_push = mem.get_possible_maps_push

        def traced_get_possible_maps_push(
            *candidate_args: Any, **candidate_kwargs: Any
        ) -> Any:
            result = original_get_possible_maps_push(
                *candidate_args, **candidate_kwargs
            )
            candidate_trace.append(candidate_set_fingerprint(result))
            return result

        mem.get_possible_maps_push = traced_get_possible_maps_push
        originals.append(
            (mem, "get_possible_maps_push", original_get_possible_maps_push)
        )

        original_eval_push_igs = mem.eval_push_igs

        def traced_eval_push_igs(*eval_args: Any, **eval_kwargs: Any) -> Any:
            result = original_eval_push_igs(*eval_args, **eval_kwargs)
            push_decisions.append(
                {
                    "candidate_index": int(result[1]),
                    "source_index": None,
                    "source_is_named": None,
                    "source_is_best_attribution_source": None,
                    "selection_utility": "unchanged_mem_push_vig",
                    "sampling_priority_provenance": "original_uniform_sampler",
                }
            )
            return result

        mem.eval_push_igs = traced_eval_push_igs
        originals.append((mem, "eval_push_igs", original_eval_push_igs))

        if initial_state_snapshot is not None:
            original_get_processed_array_and_gt_data = (
                mem.get_processed_array_and_gt_data
            )

            def traced_get_processed_array_and_gt_data(
                *processed_args: Any, **processed_kwargs: Any
            ) -> Any:
                nonlocal initial_input_hashes
                result = original_get_processed_array_and_gt_data(
                    *processed_args, **processed_kwargs
                )
                if initial_input_hashes is None:
                    with recorder.measure("initial_input_hashing"):
                        initial_input_hashes = initial_observation_hashes(
                            result[0], result[1]
                        )
                return result

            mem.get_processed_array_and_gt_data = (
                traced_get_processed_array_and_gt_data
            )
            originals.append(
                (
                    mem,
                    "get_processed_array_and_gt_data",
                    original_get_processed_array_and_gt_data,
                )
            )

        # The legacy ``pushes`` list is not step-aligned when a can-push step
        # selects neither a push nor an observation.  Observe action-call deltas
        # at the existing per-step storage boundary without changing selection.
        original_store_results = mem.store_results
        prior_action_calls = {"observation": 0, "push": 0}
        prior_decision_count = 0
        prior_execution_result_count = 0
        prior_completion_call_count = 0

        def traced_store_results(*store_args: Any, **store_kwargs: Any) -> Any:
            nonlocal prior_decision_count, prior_execution_result_count
            nonlocal prior_completion_call_count
            from shelf_gym.utils.cnabu_occlusion_planner import (
                live_cnabu_belief_arrays,
            )

            with recorder.measure("belief_diagnostics"):
                belief = live_cnabu_belief_arrays(store_args[1], store_args[2])
                belief_trace.append(
                    {
                        "occupancy_epistemic_mean": float(
                            belief["occupancy_epistemic"].mean()
                        ),
                        "semantic_vacuity_mean": float(
                            belief["semantic_vacuity"].mean()
                        ),
                    }
                )
            result = original_store_results(*store_args, **store_kwargs)
            observation_calls = int(
                recorder._records.get("observation_cnabu_update", {}).get("calls", 0)
            )
            push_calls = int(
                recorder._records.get("push_execution", {}).get("calls", 0)
            )
            observation_delta = observation_calls - prior_action_calls["observation"]
            push_delta = push_calls - prior_action_calls["push"]
            if observation_delta > 1 or push_delta > 1 or (
                observation_delta and push_delta
            ):
                raise RuntimeError("unexpected multiple action calls in one MEM step")
            step_action_trace.append(
                2 if push_delta else (0 if observation_delta else None)
            )
            completion_delta = (
                len(mapping_completion_calls) - prior_completion_call_count
            )
            if completion_delta not in (0, 1):
                raise RuntimeError("unexpected mapping-completion multiplicity")
            mapping_completion_trace.append(
                mapping_completion_calls[-1] if completion_delta else None
            )
            decision_delta = len(push_decisions) - prior_decision_count
            execution_result_delta = (
                len(push_execution_results) - prior_execution_result_count
            )
            if decision_delta not in (0, 1) or execution_result_delta not in (0, 1):
                raise RuntimeError("unexpected push diagnostic multiplicity in one step")
            if bool(push_delta) != bool(execution_result_delta):
                raise RuntimeError("push action and execution diagnostics disagree")
            selected = push_decisions[-1] if decision_delta else None
            execution = (
                push_execution_results[-1] if execution_result_delta else None
            )
            object_drop_state = bool(np.asarray(store_args[8]).any())
            object_drop = bool(push_delta and object_drop_state)
            return_code = None if execution is None else int(execution["push_return_code"])
            step_safety_trace.append(
                {
                    "attempted_push": bool(push_delta),
                    "push_return_code": return_code,
                    "tilted_object_failure": (
                        None if return_code is None else bool(return_code != 0)
                    ),
                    "object_drop": object_drop,
                    "accepted_without_tilt_or_drop": bool(
                        push_delta and return_code == 0 and not object_drop
                    ),
                    "candidate_selected_but_not_executed": bool(
                        decision_delta and not push_delta
                    ),
                    "selected_candidate": selected,
                    "cumulative_position_difference": float(store_args[5]),
                    "cumulative_position_difference_without_drop": float(
                        store_args[6]
                    ),
                }
            )
            prior_action_calls["observation"] = observation_calls
            prior_action_calls["push"] = push_calls
            prior_decision_count = len(push_decisions)
            prior_execution_result_count = len(push_execution_results)
            prior_completion_call_count = len(mapping_completion_calls)
            return result

        mem.store_results = traced_store_results
        originals.append((mem, "store_results", original_store_results))

        push_model = mem.push_prediction_model.push_predictor
        forward_started: list[float] = []

        def pre_hook(_module: Any, _inputs: Any) -> None:
            _synchronize_cuda()
            forward_started.append(time.perf_counter())

        def post_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            _synchronize_cuda()
            elapsed = time.perf_counter() - forward_started.pop()
            record = recorder._records.setdefault(
                "push_cnabu_prediction", {"calls": 0, "wall_seconds": 0.0}
            )
            record["calls"] = int(record["calls"]) + 1
            record["wall_seconds"] = float(record["wall_seconds"]) + float(elapsed)

        hook_handles.extend(
            [
                push_model.register_forward_pre_hook(pre_hook),
                push_model.register_forward_hook(post_hook),
            ]
        )

        episode_started = time.perf_counter()
        output = mem.run(
            predefined_scene_dir=str(scene_path),
            use_push=not bool(args.disable_push),
            debug=bool(args.debug),
        )
        episode_seconds = float(time.perf_counter() - episode_started)
        episode = _summarise_episode(
            output,
            n_classes=int(mem.n_classes),
            step_action_trace=step_action_trace,
            belief_trace=belief_trace,
            step_safety_trace=step_safety_trace,
            mapping_completion_trace=mapping_completion_trace,
            mapping_completion_threshold=float(mem.stopping_criterion),
        )
    finally:
        for handle in hook_handles:
            handle.remove()
        for owner, attribute, original in reversed(originals):
            _restore_callable(owner, attribute, original)
        import cupy as cp

        cp.random.choice = cupy_choice["original"]
        mem.close()

    camera_payload = np.load(DEFAULT_CAMERA_PATH, allow_pickle=True)["obj_ids"]
    summary = {
        "schema": (
            "cnabu_mem_original_baseline_smoke_v2_controlled"
            if initial_state_snapshot is not None
            else "cnabu_mem_original_baseline_smoke_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "repo": _git_state(REPO_ROOT),
        "seed": int(args.seed),
        "action_budget": int(args.action_budget),
        "max_sampled_pushes": int(args.max_sampled_pushes),
        "push_frontier_sampling_rng": {
            key: value for key, value in cupy_choice.items() if key != "original"
        },
        "push_enabled": not bool(args.disable_push),
        "render": bool(args.render),
        "experiment_control": (
            None
            if initial_state_path is None
            else {
                "policy": experiment_control,
                "process_environment": process_environment,
                "snapshot": {
                    "path": str(initial_state_path),
                    "sha256": sha256_file(initial_state_path),
                    "state_sha256": initial_state_snapshot["state_sha256"],
                },
                "application": initial_state_application,
                "initial_inputs": initial_input_hashes,
            }
        ),
        "scene": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "camera_array": {
            "path": str(DEFAULT_CAMERA_PATH.resolve()),
            "sha256": sha256_file(DEFAULT_CAMERA_PATH),
            "camera_count": int(len(camera_payload)),
        },
        "checkpoints": {
            "map_predictor": {
                "path": str(DEFAULT_MAP_CHECKPOINT.resolve()),
                "sha256": sha256_file(DEFAULT_MAP_CHECKPOINT),
            },
            "push_predictor": {
                "path": str(DEFAULT_PUSH_CHECKPOINT.resolve()),
                "sha256": sha256_file(DEFAULT_PUSH_CHECKPOINT),
            },
        },
        "timing": {
            "initialization_seconds": initialization_seconds,
            "episode_seconds": episode_seconds,
            "total_seconds": float(time.perf_counter() - started),
            "phases": recorder.as_dict(),
        },
        "memory": {
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "episode": episode,
        "candidate_generation": candidate_trace,
        "push_decisions": push_decisions,
        "safety": {
            "training_run": False,
            "checkpoint_written": False,
            "dataset_export_written": False,
            "gt_used_for_planner_input": False,
            "simulator_instance_ids_used_for_planner_input": False,
            "planner_behavior_modified": False,
            "action_trace_instrumentation_only": True,
            "common_experiment_control_enabled": initial_state_path is not None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
