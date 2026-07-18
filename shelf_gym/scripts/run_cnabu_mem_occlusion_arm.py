"""Run one fixed-budget deterministic or oracle belief-occlusion MEM arm."""

from __future__ import annotations

import argparse
import json
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict

import numpy as np

from shelf_gym.utils.cnabu_mem_experiment_control import (
    apply_initial_state_snapshot,
    candidate_set_fingerprint,
    configure_controlled_mem,
    configure_deterministic_process_environment,
    initial_observation_hashes,
    load_initial_state_snapshot,
)


_PROCESS_ENVIRONMENT = configure_deterministic_process_environment()

from scene_graph_mem.runtime.cnabu_learned_component_splitter import (
    DEFAULT_CHECKPOINT_PATH,
    LearnedCnabuComponentSplitter,
)
from shelf_gym.scripts.run_cnabu_mem_baseline_smoke import (
    DEFAULT_CAMERA_PATH,
    DEFAULT_MAP_CHECKPOINT,
    DEFAULT_PUSH_CHECKPOINT,
    PhaseRecorder,
    _configure_cupy_choice,
    _git_state,
    _restore_callable,
    _seed_everything,
    _summarise_episode,
    _synchronize_cuda,
    _wrap_callable,
    sha256_file,
)
from shelf_gym.utils.cnabu_occlusion_planner import (
    BeliefOcclusionAllocationController,
    live_cnabu_belief_arrays,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = REPO_ROOT.parent
SCENE_GRAPH_ROOT = THESIS_ROOT / "scene_graph_mem"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("deterministic", "oracle"), required=True)
    parser.add_argument("--predefined-scene", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--action-budget", type=int, default=5)
    parser.add_argument("--max-sampled-pushes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-state-snapshot", type=Path, required=True)
    parser.add_argument("--node-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    scene_path = args.predefined_scene.resolve()
    output_path = args.output_json.resolve()
    checkpoint_path = args.node_checkpoint.resolve()
    initial_state_path = args.initial_state_snapshot.resolve()
    for path in (scene_path, checkpoint_path, initial_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    if int(args.action_budget) <= 0 or int(args.max_sampled_pushes) <= 0:
        raise ValueError("action and sampled-push budgets must be positive")

    initial_state_snapshot = load_initial_state_snapshot(initial_state_path)
    _seed_everything(int(args.seed))
    import torch
    import shelf_gym.scripts.run_cnabu_pipeline as pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("the MEM checkpoints require CUDA")
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    recorder = PhaseRecorder()
    started = time.perf_counter()
    init_started = time.perf_counter()
    mem = pipeline.ManipulationEnhancedMapping(
        render=bool(args.render),
        show_vis=False,
        use_uncertainty_informed_sampling=False,
    )
    mem_initialization_seconds = float(time.perf_counter() - init_started)
    mem.action_budget = int(args.action_budget)
    mem.max_sampled_pushes = int(args.max_sampled_pushes)
    experiment_control = configure_controlled_mem(mem, seed=int(args.seed))
    splitter_started = time.perf_counter()
    splitter = LearnedCnabuComponentSplitter(
        checkpoint_path,
        device=args.device,
    )
    _synchronize_cuda()
    splitter_initialization_seconds = float(time.perf_counter() - splitter_started)
    controller = BeliefOcclusionAllocationController(
        arm=str(args.arm),
        node_splitter=splitter,
        info_gain=mem.ig_calc,
        device=str(args.device),
        guidance_fraction=0.75,
    )
    original_get_possible_maps_push = mem.get_possible_maps_push
    candidate_trace: list[Dict[str, Any]] = []
    push_decisions: list[Dict[str, Any]] = []
    push_execution_results: list[Dict[str, Any]] = []
    step_safety_trace: list[Dict[str, Any]] = []
    mapping_completion_calls: list[float] = []
    mapping_completion_trace: list[float | None] = []

    def guided_get_possible_maps_push(
        self: Any,
        beta_map: Any,
        dirichlet_map: Any,
        num_points: int = 30,
        planner_camera_index: Any = None,
        frontier_allocator: Any = None,
    ) -> Dict[str, Any]:
        if frontier_allocator is not None:
            raise ValueError("guided wrapper owns the sole frontier allocator")
        if planner_camera_index is None:
            raise RuntimeError("belief-occlusion allocation requires MEM's selected camera")
        with recorder.measure("belief_occlusion_planner"):
            allocator = controller.build_allocator(
                occupancy_distribution=beta_map,
                semantic_concentration=dirichlet_map,
                camera_index=int(planner_camera_index),
                environment=self if args.arm == "oracle" else None,
            )
        result = original_get_possible_maps_push(
            beta_map,
            dirichlet_map,
            num_points=num_points,
            planner_camera_index=planner_camera_index,
            frontier_allocator=allocator,
        )
        candidate_trace.append(candidate_set_fingerprint(result))
        paths = result.get("paths")
        controller.record_sampling_result(
            allocation_diagnostics=self.ps.last_frontier_allocation,
            feasible_path_count=0 if paths is None else len(paths),
        )
        return result

    mem.get_possible_maps_push = MethodType(guided_get_possible_maps_push, mem)

    original_eval_push_igs = mem.eval_push_igs

    def traced_eval_push_igs(*eval_args: Any, **eval_kwargs: Any) -> Any:
        result = original_eval_push_igs(*eval_args, **eval_kwargs)
        controller.record_scoring_result(selected_candidate_index=int(result[1]))
        push_decisions.append(dict(controller.history[-1]["selected_candidate"]))
        return result

    mem.eval_push_igs = traced_eval_push_igs
    cupy_choice = _configure_cupy_choice(int(args.seed), force_numpy=True)
    originals = [(mem, "eval_push_igs", original_eval_push_igs)]
    hook_handles = []
    step_action_trace: list[int | None] = []
    belief_trace: list[Dict[str, float]] = []
    initial_state_application: Dict[str, Any] | None = None
    initial_input_hashes: Dict[str, Any] | None = None
    try:
        original_restore_shelf_state = mem.restore_shelf_state

        def controlled_restore_shelf_state(
            *restore_args: Any, **restore_kwargs: Any
        ) -> Any:
            nonlocal initial_state_application
            result = original_restore_shelf_state(
                *restore_args, **restore_kwargs
            )
            initial_state_application = apply_initial_state_snapshot(
                mem,
                initial_state_snapshot,
                scene_path=scene_path,
            )
            return result

        mem.restore_shelf_state = controlled_restore_shelf_state
        originals.append((mem, "restore_shelf_state", original_restore_shelf_state))

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

        mem.get_processed_array_and_gt_data = traced_get_processed_array_and_gt_data
        originals.append(
            (
                mem,
                "get_processed_array_and_gt_data",
                original_get_processed_array_and_gt_data,
            )
        )

        original_store_results = mem.store_results
        prior_action_calls = {"observation": 0, "push": 0}
        prior_query_count = 0
        prior_decision_count = 0
        prior_execution_result_count = 0
        prior_completion_call_count = 0

        def traced_store_results(*store_args: Any, **store_kwargs: Any) -> Any:
            nonlocal prior_query_count, prior_decision_count
            nonlocal prior_execution_result_count, prior_completion_call_count
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
            query_delta = len(controller.history) - prior_query_count
            decision_delta = len(push_decisions) - prior_decision_count
            execution_result_delta = (
                len(push_execution_results) - prior_execution_result_count
            )
            if query_delta not in (0, 1) or decision_delta not in (0, 1):
                raise RuntimeError("unexpected graph push diagnostic multiplicity")
            if execution_result_delta not in (0, 1):
                raise RuntimeError("unexpected push execution diagnostic multiplicity")
            if bool(push_delta) != bool(execution_result_delta):
                raise RuntimeError("push action and execution diagnostics disagree")
            if push_delta and not query_delta:
                raise RuntimeError("guided push executed without a current graph query")
            selected = push_decisions[-1] if decision_delta else None
            execution = (
                push_execution_results[-1] if execution_result_delta else None
            )
            object_drop_state = bool(np.asarray(store_args[8]).any())
            object_drop = bool(push_delta and object_drop_state)
            return_code = None if execution is None else int(execution["push_return_code"])
            if query_delta:
                controller.record_execution_result(
                    executed=bool(push_delta),
                    push_return_code=return_code,
                    object_drop=object_drop,
                )
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
                        selected is not None and not push_delta
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
            prior_query_count = len(controller.history)
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
            use_push=True,
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
        "schema": "cnabu_mem_belief_occlusion_arm_v2_controlled",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "arm": str(args.arm),
        "repo": _git_state(REPO_ROOT),
        "scene_graph_repo": _git_state(SCENE_GRAPH_ROOT),
        "seed": int(args.seed),
        "action_budget": int(args.action_budget),
        "max_sampled_pushes": int(args.max_sampled_pushes),
        "experiment_control": {
            "policy": experiment_control,
            "process_environment": _PROCESS_ENVIRONMENT,
            "snapshot": {
                "path": str(initial_state_path),
                "sha256": sha256_file(initial_state_path),
                "state_sha256": initial_state_snapshot["state_sha256"],
            },
            "application": initial_state_application,
            "initial_inputs": initial_input_hashes,
        },
        "push_frontier_sampling_rng": {
            key: value for key, value in cupy_choice.items() if key != "original"
        },
        "frontier_allocation": {
            "guidance_fraction": 0.75,
            "uniform_fraction": 0.25,
            "without_replacement": True,
            "fixed_budget": True,
            "changes_action_utility": False,
        },
        "scene": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "camera_array": {
            "path": str(DEFAULT_CAMERA_PATH.resolve()),
            "sha256": sha256_file(DEFAULT_CAMERA_PATH),
            "camera_count": int(len(camera_payload)),
            "query_workload": "original_mem_selected_top1_unvisited_vig_camera",
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
            "learned_node_splitter": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "schema": splitter.checkpoint_schema,
            },
        },
        "timing": {
            "mem_initialization_seconds": mem_initialization_seconds,
            "node_splitter_initialization_seconds": splitter_initialization_seconds,
            "episode_seconds": episode_seconds,
            "total_seconds": float(time.perf_counter() - started),
            "phases": recorder.as_dict(),
        },
        "memory": {
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(torch.device(args.device))
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(torch.device(args.device))
            ),
            "process_max_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "episode": episode,
        "candidate_generation": candidate_trace,
        "push_decisions": push_decisions,
        "attribution_queries": controller.history,
        "safety": {
            "training_run": False,
            "checkpoint_written": False,
            "dataset_export_written": False,
            "gt_used_for_planner_input": args.arm == "oracle",
            "oracle_arm_deployable": False,
            "deterministic_arm_uses_gt": False,
            "simulator_instance_ids_used_by_deterministic_arm": False,
            "physical_relation_executed": False,
            "physical_relation_assets_or_records_loaded": False,
            "legacy_action_utility_changed": False,
            "legacy_feasibility_machinery_changed": False,
            "common_experiment_control_enabled": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "arm": str(args.arm),
                "episode_seconds": episode_seconds,
                "steps_completed": episode["steps_completed"],
                "num_executed_pushes": episode["num_executed_pushes"],
                "final_occupancy_macro_iou": episode["final_occupancy_macro_iou"],
                "final_semantic_macro_iou": episode["final_semantic_macro_iou"],
                "attribution_query_count": len(controller.history),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
