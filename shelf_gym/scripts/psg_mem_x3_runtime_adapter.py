"""Approved-run runtime adapter for live Shelf Gym PSG-MEM X3 episodes.

Importing this module is inert.  ``build_runtime_adapter`` performs the heavy
initialization only when invoked by the approval-gated X3 driver.  Runtime
inputs contain the semantic/coarse target query but never the hidden simulator
target ID or GT mask.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


RUNTIME_SPEC_SCHEMA = "shelf_gym_psg_mem_runtime_v1"
_PERSISTENT_ARMS = {"d", "f", "g", "h"}
_PRIVILEGED_RUNTIME_FIELDS = {
    "evaluation_object_id",
    "evaluation_token",
    "gt_instance_id",
    "gt_mask",
    "oracle_blockers",
    "simulator_instance_id",
    "target_mask",
}
_COMPONENT_CONFIG_FIELDS = {
    "occupancy_threshold",
    "semantic_confidence_threshold",
    "max_semantic_vacuity",
    "max_occupancy_epistemic",
    "min_voxels",
    "min_pixels",
    "connectivity",
    "object_class_max_exclusive",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(identity: Mapping[str, Any], *, name: str) -> Path:
    if not isinstance(identity, Mapping):
        raise ValueError(f"runtime integration requires {name} path/hash identity")
    raw_path = identity.get("path")
    digest = identity.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"runtime {name} path must be absolute")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest.lower())
    ):
        raise ValueError(f"runtime {name} requires a SHA-256 digest")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != digest.lower():
        raise ValueError(f"runtime {name} hash mismatch")
    return path


def _reject_privileged_runtime_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_name, child in value.items():
            name = str(raw_name)
            if name in _PRIVILEGED_RUNTIME_FIELDS:
                raise ValueError(
                    f"privileged runtime field is forbidden: {path}.{name}"
                )
            _reject_privileged_runtime_fields(child, path=f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_privileged_runtime_fields(child, path=f"{path}[{index}]")


def _process_memory(pid: int) -> Dict[str, Any]:
    """Read Linux process RSS/HWM without a sampling thread or mutation."""

    status_path = Path(f"/proc/{int(pid)}/status")
    result: Dict[str, Any] = {
        "schema": "psg_mem_process_memory_v1",
        "pid": int(pid),
        "available": False,
        "rss_bytes": None,
        "high_water_bytes": None,
    }
    try:
        values = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                fields = remainder.strip().split()
                if len(fields) == 2 and fields[1] == "kB":
                    values[name] = int(fields[0]) * 1024
        if {"VmRSS", "VmHWM"} <= set(values):
            result.update(
                {
                    "available": True,
                    "rss_bytes": int(values["VmRSS"]),
                    "high_water_bytes": int(values["VmHWM"]),
                }
            )
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return result


def validate_runtime_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError("runtime adapter specification must be a mapping")
    episode_id = spec.get("episode_id")
    arm = str(spec.get("arm_id", "")).lower()
    planner_query = spec.get("planner_query")
    integration = spec.get("integration")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("runtime adapter requires episode_id")
    if arm not in set("abcdefgh"):
        raise ValueError("runtime adapter arm must be a..h")
    if not isinstance(planner_query, Mapping):
        raise ValueError("runtime adapter requires planner_query")
    _reject_privileged_runtime_fields(planner_query, path="planner_query")
    if (
        not isinstance(integration, Mapping)
        or integration.get("schema") != RUNTIME_SPEC_SCHEMA
    ):
        raise ValueError(f"integration schema must be {RUNTIME_SPEC_SCHEMA}")
    allowed = {
        "schema",
        "scene",
        "initial_state_snapshot",
        "splitter_checkpoint",
        "relation_checkpoint",
        "relation_config",
        "reasoning_checkpoint",
        "scene_graph_python",
        "scene_graph_sidecar",
        "scene_graph_mem_src",
        "seed",
        "action_budget",
        "max_sampled_pushes",
        "render",
        "relation_device",
        "raw_shape_hw",
        "crop_rows",
        "graph_config",
        "tracker_config",
        "extractor_config",
        "runtime_graph_variant",
        "component_config",
        "sidecar_startup_timeout_seconds",
        "sidecar_request_timeout_seconds",
    }
    unexpected = sorted(set(integration) - allowed)
    if unexpected:
        raise ValueError(f"runtime integration has undeclared fields: {unexpected}")
    files = {
        name: _verified_file(integration[name], name=name)
        for name in (
            "scene",
            "initial_state_snapshot",
            "splitter_checkpoint",
            "relation_checkpoint",
            "relation_config",
            "scene_graph_python",
            "scene_graph_sidecar",
        )
    }
    if integration.get("reasoning_checkpoint") is not None:
        files["reasoning_checkpoint"] = _verified_file(
            integration["reasoning_checkpoint"], name="reasoning_checkpoint"
        )
    raw_source_root = integration.get("scene_graph_mem_src")
    if not isinstance(raw_source_root, str) or not Path(raw_source_root).is_absolute():
        raise ValueError("scene_graph_mem_src must be the absolute package source root")
    source_root = Path(raw_source_root).resolve()
    if not (source_root / "scene_graph_mem").is_dir():
        raise ValueError("scene_graph_mem_src must be the absolute package source root")
    expected_source_root = (
        files["scene_graph_sidecar"].parent.parent / "src"
    ).resolve()
    if source_root != expected_source_root:
        raise ValueError(
            "scene_graph_mem_src must belong to the hash-pinned sidecar repository"
        )
    for name in ("seed", "action_budget", "max_sampled_pushes"):
        value = integration.get(name)
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"runtime {name} must be an integer >= {minimum}")
    if not isinstance(integration.get("render"), bool):
        raise ValueError("runtime render must be boolean")
    if integration.get("relation_device") not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("runtime relation_device must be cpu/cuda/cuda:0")
    raw_shape = tuple(int(value) for value in integration.get("raw_shape_hw", ()))
    crop_rows = tuple(int(value) for value in integration.get("crop_rows", ()))
    if raw_shape != (140, 200) or crop_rows != (10, 130):
        raise ValueError(
            "live MEM graph coordinates require raw [140,200], crop [10,130]"
        )
    for name in (
        "graph_config",
        "tracker_config",
        "extractor_config",
        "runtime_graph_variant",
        "component_config",
    ):
        value = integration.get(name, {})
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"runtime {name} must be a mapping")
        _reject_privileged_runtime_fields(value or {}, path=f"integration.{name}")
    component_config = dict(integration.get("component_config") or {})
    unexpected_component = sorted(set(component_config) - _COMPONENT_CONFIG_FIELDS)
    if unexpected_component:
        raise ValueError(
            "runtime component_config has undeclared fields: "
            f"{unexpected_component}"
        )
    runtime_variant = dict(integration.get("runtime_graph_variant") or {})
    if runtime_variant:
        if runtime_variant.get("schema") != "psg_mem_runtime_graph_variant_v1":
            raise ValueError("runtime_graph_variant has an unexpected schema")
        family = runtime_variant.get("family")
        allowed_by_family = {
            "x5_confidence": {
                "schema",
                "family",
                "mode",
                "binary_threshold",
            },
            "x8_relation_perturbation": {
                "schema",
                "family",
                "mode",
                "seed",
                "sigma",
                "drop_rate",
            },
        }
        if family not in allowed_by_family:
            raise ValueError("runtime_graph_variant has an unexpected family")
        unexpected_variant = sorted(
            set(runtime_variant) - allowed_by_family[str(family)]
        )
        if unexpected_variant:
            raise ValueError(
                "runtime_graph_variant has undeclared fields: "
                f"{unexpected_variant}"
            )
    extractor_config = dict(integration.get("extractor_config") or {})
    if extractor_config.get("write_snapshots") not in (None, False):
        raise ValueError("live runtime may not write graph snapshots")
    for name in (
        "sidecar_startup_timeout_seconds",
        "sidecar_request_timeout_seconds",
    ):
        value = integration.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"runtime {name} must be a positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"runtime {name} must be a positive number")
    return {
        "episode_id": episode_id,
        "arm_id": arm,
        "planner_query": dict(planner_query),
        "integration": dict(integration),
        "files": files,
        "scene_graph_mem_src": source_root,
        "persistent": arm in _PERSISTENT_ARMS,
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import cupy as cp
    import torch

    cp.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _finish_runtime_adapter(
    *,
    validated: Mapping[str, Any],
    integration: Mapping[str, Any],
    process_environment: Mapping[str, Any],
    experiment_control: Mapping[str, Any],
    cupy_choice: Mapping[str, Any],
    mem: Any,
    pipeline: Any,
    torch: Any,
    cleanup: ExitStack,
) -> Dict[str, Any]:
    from shelf_gym.utils.action_conditioned_relation_oracle import (
        build_cnabu_runtime_candidate_action_mask,
    )
    from shelf_gym.utils.cnabu_mem_experiment_control import (
        apply_initial_state_snapshot,
        load_initial_state_snapshot,
    )
    from shelf_gym.utils.cnabu_occlusion_planner import live_cnabu_belief_arrays
    from shelf_gym.utils.psg_mem_action_adapter import (
        MemActionExecutionState,
        OfficialMemActionExecutor,
    )
    from shelf_gym.utils.psg_mem_live_bridge import OfficialMemStepBridge
    from shelf_gym.utils.psg_mem_graph_sidecar import PsgMemGraphSidecarClient
    from shelf_gym.utils.psg_mem_live_registry import (
        LiveEpisodeHandle,
        close_live_episode,
        register_live_episode,
    )
    from scene_graph_mem.memory.graph_tracker import GraphTracker, PushUpdateContext
    from scene_graph_mem.experiments.graph_variants import (
        apply_runtime_graph_variant,
    )
    from scene_graph_mem.runtime.cnabu_candidate_trajectory import (
        CnabuRuntimeSupportNodes,
        reconstruct_cnabu_candidate_trajectory_support,
    )
    from scene_graph_mem.runtime.mem_candidate_bridge import bind_mem_action_candidates

    scene_path = validated["files"]["scene"]
    with scene_path.open("rb") as handle:
        arrangement = pickle.load(handle)
    mem.restore_shelf_state(arrangement)
    snapshot = load_initial_state_snapshot(validated["files"]["initial_state_snapshot"])
    initial_state_application = apply_initial_state_snapshot(
        mem, snapshot, scene_path=scene_path
    )
    # Preserve the official initial camera-array query while explicitly
    # suppressing the baseline's unused GT return on the runtime side.
    _camera_data, empty_gt = mem.get_processed_array_and_gt_data(only_array=True)
    if empty_gt:
        raise RuntimeError("runtime-only initial observation unexpectedly returned GT")
    initial = torch.ones((1, 1, 204, 120, 200), device="cuda")
    previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
        initial
    )
    state = MemActionExecutionState(
        previous_map=previous_map,
        previous_semantic_map=previous_semantic_map,
    )
    action_executor = OfficialMemActionExecutor(
        mem,
        state=state,
        execute_push=pipeline.execute_push,
    )

    context_source = Path(
        sys.modules["shelf_gym.utils.action_conditioned_relation_oracle"].__file__
    ).resolve()
    candidate_context_identity = {
        "schema": "cnabu_live_candidate_context_v1",
        "path": str(context_source),
        "source_sha256": _sha256(context_source),
        "transport": "psg_mem_graph_sidecar_v1",
    }

    def candidate_context_provider(**kwargs: Any) -> Dict[str, Any]:
        runtime_graph = kwargs["runtime_graph"]
        raw_nodes = list(runtime_graph.get("nodes") or [])
        if not raw_nodes:
            raise ValueError("learned sidecar returned no runtime nodes")
        nodes = CnabuRuntimeSupportNodes.from_runtime_graph(runtime_graph)
        component_contract = copy.deepcopy(
            dict(
                (runtime_graph.get("thresholds") or {}).get(
                    "component_extraction"
                )
                or kwargs.get("component_config")
                or {}
            )
        )
        support = reconstruct_cnabu_candidate_trajectory_support(
            occupancy_mean=kwargs["occupancy_mean"],
            semantic_mean=kwargs["semantic_mean"],
            occupancy_epistemic=kwargs.get("occupancy_epistemic"),
            semantic_vacuity=kwargs.get("semantic_vacuity"),
            nodes=nodes,
            crop_rows=kwargs["crop_rows"],
            reconstruction_kwargs=component_contract,
        )
        action_mask = build_cnabu_runtime_candidate_action_mask(
            mem,
            mem.smg.hg,
            support.indices_zyx,
            crop_rows=kwargs["crop_rows"],
            node_ids=nodes.node_ids,
            initial_arm_config=np.asarray(
                mem.get_current_joint_config(), dtype=np.float64
            ),
            support_boundary_quantile=0.05,
            include_planner_swept_features=True,
        )
        return {
            "candidate_action_mask": action_mask,
            "candidate_planner_swept_features": action_mask["planner_swept_features"],
            "metadata": {
                "uses_gt": False,
                "uses_simulator_instance_ids": False,
                "source": "live_cnabu_support_current_robot_fixed_environment_v1",
            },
        }

    graph_sidecar = PsgMemGraphSidecarClient(
        python=validated["files"]["scene_graph_python"],
        sidecar_script=validated["files"]["scene_graph_sidecar"],
        splitter_checkpoint=validated["files"]["splitter_checkpoint"],
        relation_checkpoint=validated["files"]["relation_checkpoint"],
        relation_config=validated["files"]["relation_config"],
        reasoning_checkpoint=validated["files"].get("reasoning_checkpoint"),
        device=str(integration["relation_device"]),
        seed=int(integration["seed"]),
        graph_config=dict(integration.get("graph_config") or {}),
        extractor_config=dict(integration.get("extractor_config") or {}),
        candidate_context_identity=candidate_context_identity,
        startup_timeout_seconds=float(integration["sidecar_startup_timeout_seconds"]),
        request_timeout_seconds=float(integration["sidecar_request_timeout_seconds"]),
    )
    cleanup.callback(graph_sidecar.close)
    tracker = (
        GraphTracker(dict(integration.get("tracker_config") or {}))
        if validated["persistent"]
        else None
    )

    def graph_provider(**kwargs: Any) -> Dict[str, Any]:
        provider_started = time.perf_counter()
        parent_memory_before = _process_memory(os.getpid())
        cuda_available = bool(torch.cuda.is_available())
        cuda_allocated_before = (
            int(torch.cuda.memory_allocated()) if cuda_available else 0
        )
        cuda_peak_before = (
            int(torch.cuda.max_memory_allocated()) if cuda_available else 0
        )
        belief_started = time.perf_counter()
        beliefs = live_cnabu_belief_arrays(
            kwargs["occupancy_distribution"],
            kwargs["semantic_concentration"],
        )
        belief_seconds = time.perf_counter() - belief_started
        sidecar_started = time.perf_counter()
        extracted = graph_sidecar.extract(
            episode_id=kwargs["episode_id"],
            step=int(kwargs["step"]),
            occupancy_mean=beliefs["occupancy_mean"],
            semantic_mean=beliefs["semantic_mean"],
            occupancy_variance=beliefs["occupancy_epistemic"],
            semantic_vacuity=beliefs["semantic_vacuity"],
            raw_shape_hw=(140, 200),
            crop_rows=(10, 130),
            target_query=kwargs["target_query"],
            selected_view_indices=kwargs["selected_view_indices"],
            component_config=dict(integration.get("component_config") or {}),
            candidate_context_provider=candidate_context_provider,
            metadata={
                "source": "official_mem_live_belief",
                "uses_gt": False,
                "uses_simulator_instance_ids": False,
            },
        )
        sidecar_seconds = time.perf_counter() - sidecar_started
        variant_started = time.perf_counter()
        runtime_variant = dict(integration.get("runtime_graph_variant") or {})
        if runtime_variant:
            extracted = apply_runtime_graph_variant(extracted, runtime_variant)
        variant_seconds = time.perf_counter() - variant_started
        tracker_started = time.perf_counter()
        if tracker is None:
            extracted["metadata"]["persistent_memory"] = False
            result = extracted
        else:
            previous_execution = kwargs.get("previous_execution") or {}
            selected_action = previous_execution.get("selected_action") or {}
            push_context = None
            if selected_action.get("kind") == "push":
                direction = selected_action.get("push_direction_xy")
                push_context = PushUpdateContext(
                    pushed_node_id=selected_action.get("source_node_id"),
                    direction_xy=(
                        None
                        if direction is None
                        else tuple(float(value) for value in direction)
                    ),
                    action_id=selected_action.get("candidate_fingerprint"),
                )
            result = tracker.update(extracted, push_context=push_context)
        tracker_seconds = time.perf_counter() - tracker_started
        parent_memory_after = _process_memory(os.getpid())
        cuda_allocated_after = (
            int(torch.cuda.memory_allocated()) if cuda_available else 0
        )
        cuda_peak_after = (
            int(torch.cuda.max_memory_allocated()) if cuda_available else 0
        )
        total_seconds = time.perf_counter() - provider_started
        result["metadata"]["runtime_pipeline_timing_seconds"] = {
            "belief_conversion": float(belief_seconds),
            "sidecar_transport_and_extraction": float(sidecar_seconds),
            "runtime_graph_variant": float(variant_seconds),
            "tracking": float(tracker_seconds),
            "total_graph_provider": float(total_seconds),
        }
        resources = copy.deepcopy(
            dict(result["metadata"].get("runtime_resources") or {})
        )
        resources["parent_process"] = {
            "before": parent_memory_before,
            "after": parent_memory_after,
            "rss_delta_bytes": (
                None
                if not parent_memory_before["available"]
                or not parent_memory_after["available"]
                else int(
                    parent_memory_after["rss_bytes"]
                    - parent_memory_before["rss_bytes"]
                )
            ),
        }
        resources["parent_cuda"] = {
            "available": cuda_available,
            "allocated_before_bytes": cuda_allocated_before,
            "allocated_after_bytes": cuda_allocated_after,
            "allocated_delta_bytes": int(
                cuda_allocated_after - cuda_allocated_before
            ),
            "peak_before_bytes": cuda_peak_before,
            "peak_after_bytes": cuda_peak_after,
            "peak_delta_bytes": int(cuda_peak_after - cuda_peak_before),
        }
        result["metadata"]["runtime_resources"] = resources
        return result

    bridge = OfficialMemStepBridge(
        mem,
        state=state,
        action_executor=action_executor,
        pipeline_module=pipeline,
        graph_provider=graph_provider,
        candidate_binder=lambda graph, candidates: bind_mem_action_candidates(
            graph, candidates, maximum_contact_distance_pixels=12.0
        ),
        episode_id=validated["episode_id"],
        target_query=validated["planner_query"],
        config={
            "enabled": True,
            "action_budget": int(integration["action_budget"]),
            "use_push": True,
            "first_push_step": 3,
            "treatment_first_push_step": 1,
            "reserve_final_observation": True,
            "map_width": 200,
            "crop_row_offset": 10,
        },
    )
    live_handle = LiveEpisodeHandle(
        episode_id=validated["episode_id"],
        mem=mem,
        bridge=bridge,
        scene_path=str(scene_path),
    )
    register_live_episode(live_handle)
    cleanup.callback(close_live_episode, validated["episode_id"])

    def step_provider(**kwargs: Any):
        payload = bridge.step_provider(**kwargs)
        live_handle.latest_graph = payload["graph"]
        return payload

    provenance = {
        "schema": "shelf_gym_psg_mem_runtime_provenance_v1",
        "episode_id": validated["episode_id"],
        "arm_id": validated["arm_id"],
        "seed": int(integration["seed"]),
        "persistent_memory": bool(validated["persistent"]),
        "process_environment": dict(process_environment),
        "experiment_control": dict(experiment_control),
        "cupy_choice": {
            key: value for key, value in cupy_choice.items() if key != "original"
        },
        "files": {
            name: {
                "path": str(path),
                "sha256": str(integration[name]["sha256"]).lower(),
            }
            for name, path in validated["files"].items()
        },
        "scene_graph_mem_src": str(validated["scene_graph_mem_src"]),
        "initial_state_application": initial_state_application,
        "extractor_config_hash": graph_sidecar.ready_provenance[
            "extractor_config_hash"
        ],
        "graph_sidecar": graph_sidecar.ready_provenance,
        "runtime_graph_variant": copy.deepcopy(
            dict(integration.get("runtime_graph_variant") or {})
        ),
        "component_config": copy.deepcopy(
            dict(integration.get("component_config") or {})
        ),
        "runtime_input_safety": {
            "uses_gt": False,
            "uses_simulator_instance_ids": False,
            "initial_observation_only_array": True,
            "writes_graph_snapshots": False,
            "writes_checkpoints_or_models": False,
        },
    }
    json.dumps(provenance, allow_nan=False)
    return {
        "step_provider": step_provider,
        "execute_mem_action": bridge.execute_mem_action,
        "close": cleanup.close,
        "provenance": provenance,
    }


def build_runtime_adapter(spec: Mapping[str, Any]) -> Dict[str, Any]:
    validated = validate_runtime_spec(spec)
    integration = validated["integration"]
    source_root = validated["scene_graph_mem_src"]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from shelf_gym.utils.cnabu_mem_experiment_control import (
        configure_controlled_mem,
        configure_deterministic_process_environment,
    )

    # This must precede Torch and the CUDA-backed MEM pipeline imports.
    process_environment = configure_deterministic_process_environment()

    import cupy as cp
    import torch
    import shelf_gym.scripts.run_cnabu_pipeline as pipeline
    from shelf_gym.scripts.run_cnabu_mem_baseline_smoke import _configure_cupy_choice

    if not torch.cuda.is_available():
        raise RuntimeError("live official MEM execution requires CUDA")
    seed = int(integration["seed"])
    _seed_everything(seed)
    cleanup = ExitStack()
    try:
        mem = pipeline.ManipulationEnhancedMapping(
            render=bool(integration["render"]),
            show_vis=False,
            use_uncertainty_informed_sampling=False,
        )
        cleanup.callback(mem.close)
        mem.action_budget = int(integration["action_budget"])
        mem.max_sampled_pushes = int(integration["max_sampled_pushes"])
        experiment_control = configure_controlled_mem(mem, seed=seed)
        cupy_choice = _configure_cupy_choice(seed, force_numpy=True)
        cleanup.callback(setattr, cp.random, "choice", cupy_choice["original"])
        return _finish_runtime_adapter(
            validated=validated,
            integration=integration,
            process_environment=process_environment,
            experiment_control=experiment_control,
            cupy_choice=cupy_choice,
            mem=mem,
            pipeline=pipeline,
            torch=torch,
            cleanup=cleanup,
        )
    except BaseException:
        cleanup.close()
        raise


__all__ = ["RUNTIME_SPEC_SCHEMA", "build_runtime_adapter", "validate_runtime_spec"]
