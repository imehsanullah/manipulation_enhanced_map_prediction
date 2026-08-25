"""Default-off MEM action boundary for an external PSG-MEM decision provider.

The official policy remains the caller's default.  This adapter is invoked
only by an explicitly enabled treatment arm, verifies that the returned
action belongs to the frozen candidate set, and then calls the injected MEM
executor exactly once.  Graph inference may run in the existing
``scene_graph_mem`` sidecar process; no graph dependency is imported here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np


_OFFICIAL_SCORING_CAPTURE_LOCK = threading.RLock()


def _candidate_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("candidate IDs must be strings or integers")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _host_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _array_identity(value: Any) -> Dict[str, Any]:
    array = np.ascontiguousarray(_host_array(value))
    if array.dtype.hasobject:
        raise ValueError("MEM candidate arrays must not use object dtype")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _push_candidate_fingerprint(push_data: Mapping[str, Any], index: int) -> str:
    paths = push_data.get("paths")
    annotations = push_data.get("path_annotations")
    motion = _host_array(push_data.get("motion_parametrization"))
    if paths is None or annotations is None:
        raise ValueError("push payload is missing paths or annotations")
    if not 0 <= int(index) < len(paths) or len(annotations) != len(paths):
        raise ValueError("push candidate index is outside the aligned payload")
    if motion.ndim != 2 or motion.shape != (len(paths), 6):
        raise ValueError("MEM motion_parametrization must have shape [N,6]")
    payload = {
        "path": _array_identity(paths[int(index)]),
        "path_annotations": copy.deepcopy(annotations[int(index)]),
        "motion_parametrization": _array_identity(motion[int(index)]),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_frozen_mem_action_candidates(
    push_data: Mapping[str, Any],
    *,
    viewpoint: int,
    viewpoint_information_gain: float,
    push_information_gains: Sequence[float],
    push_feasibilities: Sequence[float] | None = None,
    collision_free: Sequence[bool] | None = None,
    map_width: int = 200,
    crop_row_offset: int = 10,
    include_noop: bool = False,
) -> list[Dict[str, Any]]:
    """Encode one official MEM view and its exact feasible push set.

    Actions are small immutable tokens.  The physical paths stay inside the
    official MEM process and are recovered by index only after the token's
    content fingerprint is verified.  ``contact_start_yx`` follows the same
    conversion as CNABU's frozen ``add_push_features`` implementation:
    ``[motion[0] + crop_row_offset, map_width - motion[1]]``.  The production
    defaults map the 120-row MEM crop into raw 140-row graph coordinates.
    """

    if isinstance(viewpoint, bool) or not isinstance(viewpoint, (int, np.integer)):
        raise ValueError("MEM viewpoint must be an integer")
    view_ig = float(viewpoint_information_gain)
    if not math.isfinite(view_ig):
        raise ValueError("viewpoint information gain must be finite")
    if isinstance(map_width, bool) or not isinstance(map_width, int) or map_width < 1:
        raise ValueError("map_width must be a positive integer")
    if (
        isinstance(crop_row_offset, bool)
        or not isinstance(crop_row_offset, int)
        or crop_row_offset < 0
    ):
        raise ValueError("crop_row_offset must be a non-negative integer")
    if not isinstance(include_noop, bool):
        raise ValueError("include_noop must be boolean")
    result = [
        {
            "candidate_id": f"view:{int(viewpoint)}",
            "candidate_type": "view",
            "source_node_id": None,
            "feasibility": 1.0,
            "information_gain": view_ig,
            "valid": True,
            "collision_free": True,
            "action": {"kind": "observe", "viewpoint": int(viewpoint)},
        }
    ]
    paths = push_data.get("paths")
    if paths is None:
        if list(push_information_gains):
            raise ValueError("push scores were supplied for an empty candidate set")
        if include_noop:
            result.append(
                {
                    "candidate_id": "noop:official_policy",
                    "candidate_type": "noop",
                    "source_node_id": None,
                    "feasibility": 1.0,
                    "information_gain": 0.0,
                    "valid": True,
                    "collision_free": True,
                    "action": {
                        "kind": "noop",
                        "reason": "push_not_better_than_view_horizon",
                    },
                }
            )
        return result
    annotations = push_data.get("path_annotations")
    if annotations is None or len(annotations) != len(paths):
        raise ValueError("MEM push paths and annotations must align")
    motion = _host_array(push_data.get("motion_parametrization"))
    if motion.ndim != 2 or motion.shape != (len(paths), 6):
        raise ValueError("MEM motion_parametrization must have shape [N,6]")
    scores = [float(value) for value in push_information_gains]
    if len(scores) != len(paths) or not all(math.isfinite(value) for value in scores):
        raise ValueError("one finite information-gain score is required per push")
    feasibility = (
        [1.0] * len(paths)
        if push_feasibilities is None
        else [float(value) for value in push_feasibilities]
    )
    if len(feasibility) != len(paths) or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in feasibility
    ):
        raise ValueError("push feasibility values must align and lie in [0,1]")
    safe = [True] * len(paths) if collision_free is None else list(collision_free)
    if len(safe) != len(paths) or any(not isinstance(value, bool) for value in safe):
        raise ValueError("collision_free must contain one boolean per push")
    for index in range(len(paths)):
        fingerprint = _push_candidate_fingerprint(push_data, index)
        contact_end_yx = [
            float(motion[index, 2] + int(crop_row_offset)),
            float(int(map_width) - motion[index, 3]),
        ]
        push_direction_xy = [
            float(contact_end_yx[1] - (int(map_width) - motion[index, 1])),
            float(contact_end_yx[0] - (motion[index, 0] + int(crop_row_offset))),
        ]
        result.append(
            {
                "candidate_id": f"push:{index}",
                "candidate_type": "push",
                "contact_start_yx": [
                    float(motion[index, 0] + int(crop_row_offset)),
                    float(int(map_width) - motion[index, 1]),
                ],
                "contact_end_yx": contact_end_yx,
                "push_direction_xy": push_direction_xy,
                "contact_coordinate_frame": {
                    "name": "raw_heightmap_yx",
                    "source": "mem_cropped_motion_parametrization_v1",
                    "crop_row_offset": int(crop_row_offset),
                    "map_width": int(map_width),
                },
                "feasibility": float(feasibility[index]),
                "information_gain": float(scores[index]),
                "valid": True,
                "collision_free": bool(safe[index]),
                "action": {
                    "kind": "push",
                    "candidate_index": int(index),
                    "candidate_fingerprint": fingerprint,
                    "push_direction_xy": push_direction_xy,
                },
            }
        )
    if include_noop:
        result.append(
            {
                "candidate_id": "noop:official_policy",
                "candidate_type": "noop",
                "source_node_id": None,
                "feasibility": 1.0,
                "information_gain": 0.0,
                "valid": True,
                "collision_free": True,
                "action": {
                    "kind": "noop",
                    "reason": "push_not_better_than_view_horizon",
                },
            }
        )
    json.dumps(result, allow_nan=False)
    return result


def score_official_mem_push_candidates(
    mem: Any,
    pipeline_module: Any,
    push_data: Mapping[str, Any],
    *,
    previous_semantic_map: Any,
    use_delta_H: bool = True,
    skip: int = 5,
) -> Dict[str, Any]:
    """Capture per-candidate coarse IGs during the unchanged MEM scorer call.

    ``ManipulationEnhancedMapping.eval_push_igs`` computes every candidate's
    coarse push-view IG but returns only the winner.  The bridge temporarily
    observes those existing calls, restores the module function in ``finally``,
    and never performs a second scoring pass.  This keeps the official winner
    and comparison value under official code authority while exposing the
    same frozen candidate utilities to treatment tie-breaking.
    """

    if not isinstance(use_delta_H, bool):
        raise ValueError("use_delta_H must be boolean")
    if isinstance(skip, bool) or not isinstance(skip, int) or skip < 1:
        raise ValueError("skip must be a positive integer")
    paths = push_data.get("paths")
    if paths is None or not paths:
        raise ValueError("official push scoring requires a non-empty path set")
    method = getattr(mem, "eval_push_igs", None)
    original = getattr(pipeline_module, "get_igs_for_map", None)
    if not callable(method) or not callable(original):
        raise TypeError("MEM scorer and pipeline get_igs_for_map must be callable")
    push_calculator = getattr(mem, "push_ig_calc", None)
    captured: list[float] = []

    def traced(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calculator = args[1] if len(args) >= 2 else kwargs.get("ig_calc")
        if calculator is push_calculator:
            values = _host_array(result[0]).astype(np.float64, copy=False)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(
                    "official candidate IG output must be finite and non-empty"
                )
            captured.append(float(values.max()))
        return result

    with _OFFICIAL_SCORING_CAPTURE_LOCK:
        setattr(pipeline_module, "get_igs_for_map", traced)
        try:
            official = method(
                push_data,
                previous_semantic_map,
                use_delta_H=bool(use_delta_H),
                skip=int(skip),
            )
        finally:
            setattr(pipeline_module, "get_igs_for_map", original)
    if not isinstance(official, Sequence) or len(official) != 3:
        raise TypeError(
            "official eval_push_igs must return view/index/information gain"
        )
    if len(captured) != len(paths):
        raise RuntimeError(
            "official push IG capture did not align one-to-one with feasible paths"
        )
    view, candidate_index, best_information_gain = official
    candidate_index = int(candidate_index)
    best_information_gain = float(best_information_gain)
    if (
        candidate_index < 0
        or candidate_index >= len(paths)
        or not math.isfinite(best_information_gain)
    ):
        raise ValueError("official push scorer returned an invalid winner")
    result = {
        "schema": "official_mem_push_candidate_scores_v1",
        "official_viewpoint_after_push": int(view),
        "official_candidate_index": candidate_index,
        "official_push_information_gain": best_information_gain,
        "candidate_information_gains": captured,
        "candidate_score_semantics": "unchanged_mem_push_ig_calc_max_before_delta_H",
        "official_winner_semantics": "unchanged_mem_eval_push_igs_return",
        "extra_scoring_passes": 0,
        "module_function_restored": getattr(pipeline_module, "get_igs_for_map")
        is original,
    }
    json.dumps(result, allow_nan=False)
    return result


@dataclass
class MemActionExecutionState:
    """Mutable belief/action state owned by one live official MEM episode."""

    previous_map: Any
    previous_semantic_map: Any
    previous_views: list[Any] = field(default_factory=list)
    fresh_push: bool = False
    collision: bool = False
    action_count: int = 0
    last_action: Dict[str, Any] | None = None
    pushed_candidate_index: int | None = None


class OfficialMemActionExecutor:
    """Execute a verified action token through existing MEM methods once."""

    def __init__(
        self,
        mem: Any,
        *,
        state: MemActionExecutionState,
        execute_push: Callable[..., Any],
    ) -> None:
        if not isinstance(state, MemActionExecutionState):
            raise TypeError("state must be MemActionExecutionState")
        if not callable(execute_push):
            raise TypeError("execute_push must be callable")
        if not isinstance(state.previous_views, list):
            raise TypeError("state.previous_views must be a list")
        self.mem = mem
        self.state = state
        self.execute_push = execute_push
        self._push_data: Mapping[str, Any] | None = None

    def set_push_candidates(self, push_data: Mapping[str, Any]) -> None:
        if not isinstance(push_data, Mapping):
            raise TypeError("push_data must be a mapping")
        paths = push_data.get("paths")
        if paths is not None:
            for index in range(len(paths)):
                _push_candidate_fingerprint(push_data, index)
        self._push_data = push_data

    def execute(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, Mapping):
            raise TypeError("MEM action token must be a mapping")
        token = copy.deepcopy(dict(action))
        kind = token.get("kind")
        if kind in {"observe", "view"}:
            if self.state.collision:
                raise RuntimeError(
                    "cannot observe after a terminal object-drop collision"
                )
            viewpoint = token.get("viewpoint", token.get("view"))
            if isinstance(viewpoint, bool) or not isinstance(
                viewpoint, (int, np.integer)
            ):
                raise ValueError("observation action requires an integer viewpoint")
            method = getattr(self.mem, "execute_observation", None)
            if not callable(method):
                raise TypeError("MEM object does not expose execute_observation")
            updated = method(
                self.state.previous_views,
                int(viewpoint),
                self.state.previous_map,
                self.state.previous_semantic_map,
            )
            if not isinstance(updated, Sequence) or len(updated) != 2:
                raise TypeError("execute_observation must return two belief maps")
            self.state.previous_map, self.state.previous_semantic_map = updated
            self.state.fresh_push = False
            result = {
                "schema": "official_mem_action_execution_v1",
                "action_kind": "observe",
                "viewpoint": int(viewpoint),
                "collision": False,
                "executes_action": True,
            }
        elif kind == "noop":
            if self.state.collision:
                raise RuntimeError("cannot consume a no-op after a terminal collision")
            if self.state.fresh_push:
                raise RuntimeError("official MEM requires an observation after a push")
            result = {
                "schema": "official_mem_action_execution_v1",
                "action_kind": "noop",
                "reason": str(token.get("reason", "official_policy_no_action")),
                "collision": False,
                "executes_action": False,
                "action_budget_consumed": True,
            }
        elif kind == "push":
            if self.state.collision:
                raise RuntimeError("cannot push after a terminal object-drop collision")
            if self.state.fresh_push:
                raise RuntimeError("official MEM requires an observation after a push")
            if self._push_data is None or self._push_data.get("paths") is None:
                raise ValueError("no live MEM push candidate set is installed")
            index = token.get("candidate_index")
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("push action requires an integer candidate_index")
            index = int(index)
            expected = _push_candidate_fingerprint(self._push_data, index)
            if token.get("candidate_fingerprint") != expected:
                raise ValueError("stale or mismatched MEM candidate fingerprint")
            paths = self._push_data["paths"]
            annotations = self._push_data["path_annotations"]
            raw_result = self.execute_push(
                self.mem,
                paths[index],
                path_annotations=annotations[index],
            )
            if not isinstance(raw_result, Sequence) or not raw_result:
                raise TypeError(
                    "official execute_push must return a non-empty sequence"
                )
            return_code = int(raw_result[0])
            objects = getattr(self.mem, "obj", None)
            drop_check = getattr(objects, "check_all_object_drop", None)
            if not callable(drop_check):
                raise TypeError("MEM object does not expose object-drop checking")
            collision = bool(drop_check(self.mem.current_obj_ids))
            self.state.collision = collision
            self.state.fresh_push = not collision
            self.state.pushed_candidate_index = index
            if not collision:
                self.state.previous_views.clear()
                self.state.previous_map = self._push_data["possible_previous_maps"][
                    index
                ][None]
                self.state.previous_semantic_map = self._push_data[
                    "possible_semantic_maps"
                ][index][None]
            result = {
                "schema": "official_mem_action_execution_v1",
                "action_kind": "push",
                "candidate_index": index,
                "candidate_fingerprint": expected,
                "push_return_code": return_code,
                "tilted_object_failure": bool(return_code != 0),
                "collision": collision,
                "executes_action": True,
            }
        else:
            raise ValueError(f"unsupported MEM action kind: {kind!r}")
        self.state.action_count += 1
        self.state.last_action = copy.deepcopy(token)
        result["action_count"] = int(self.state.action_count)
        json.dumps(result, allow_nan=False)
        return result


@dataclass(frozen=True)
class PsgMemActionAdapterConfig:
    enabled: bool = False
    require_offered_candidate: bool = True
    allow_official_fallback: bool = True


class PsgMemActionAdapter:
    """Validate a PSG-MEM decision and execute it through official MEM code."""

    def __init__(
        self,
        decision_provider: Callable[..., Mapping[str, Any]],
        *,
        config: PsgMemActionAdapterConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if config is None:
            self.config = PsgMemActionAdapterConfig()
        elif isinstance(config, PsgMemActionAdapterConfig):
            self.config = config
        elif isinstance(config, Mapping):
            self.config = PsgMemActionAdapterConfig(**dict(config))
        else:
            raise TypeError("action adapter config must be a mapping or dataclass")
        for name in ("enabled", "require_offered_candidate", "allow_official_fallback"):
            if not isinstance(getattr(self.config, name), bool):
                raise ValueError(f"action adapter {name} must be boolean")
        if not callable(decision_provider):
            raise TypeError("decision_provider must be callable")
        self.decision_provider = decision_provider

    def step(
        self,
        *,
        graph: Mapping[str, Any],
        push_candidates: Sequence[Mapping[str, Any]],
        official_action: Mapping[str, Any] | None,
        execute_mem_action: Callable[[Any], Any],
        arm_id: str,
        target_query: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return inert delegation when off, or execute one offered action."""

        if not bool(self.config.enabled):
            return {
                "schema": "mem_psg_action_adapter_result_v1",
                "enabled": False,
                "delegates_to_official_caller": True,
                "decision_provider_called": False,
                "executes_action": False,
                "official_policy_untouched": True,
            }
        if not callable(execute_mem_action):
            raise TypeError("execute_mem_action must be callable")
        allowed_query_fields = {"class_id", "class_probs", "coarse_region"}
        unexpected_query_fields = sorted(set(target_query) - allowed_query_fields)
        if unexpected_query_fields:
            raise ValueError(
                "planner target query contains privileged or undeclared fields: "
                f"{unexpected_query_fields}"
            )
        if "class_id" not in target_query and "class_probs" not in target_query:
            raise ValueError("planner target query requires class_id or class_probs")
        candidates = [copy.deepcopy(dict(row)) for row in push_candidates]
        if not candidates:
            raise ValueError(
                "PSG-MEM treatment requires a non-empty frozen candidate set"
            )
        candidate_by_id = {}
        for row in candidates:
            candidate_id = row.get("candidate_id")
            key = _candidate_key(candidate_id)
            if key in candidate_by_id:
                raise ValueError("push candidate IDs must be present and unique")
            candidate_by_id[key] = row
        decision = self.decision_provider(
            arm_id=str(arm_id),
            graph=copy.deepcopy(dict(graph)),
            push_candidates=candidates,
            official_action=(
                None
                if official_action is None
                else copy.deepcopy(dict(official_action))
            ),
            target_query=copy.deepcopy(dict(target_query)),
        )
        if not isinstance(decision, Mapping):
            raise TypeError("decision provider must return a mapping")
        decision = copy.deepcopy(dict(decision))
        if bool(decision.get("executes_action", False)):
            raise ValueError(
                "sidecar decision must not claim it already executed an action"
            )
        selected_action = decision.get("selected_action")
        if selected_action is None:
            raise ValueError("sidecar decision does not contain an executable action")
        candidate_id = decision.get("selected_candidate_id")
        offered = (
            None
            if candidate_id is None
            else candidate_by_id.get(_candidate_key(candidate_id))
        )
        is_official_fallback = decision.get("decision_source") in {
            "official_fallback",
            "official_policy",
        }
        if (
            bool(self.config.require_offered_candidate)
            and offered is None
            and not (is_official_fallback and bool(self.config.allow_official_fallback))
        ):
            raise ValueError(
                "PSG-MEM selected an action outside the frozen candidate set"
            )
        if offered is not None and offered.get("action") != selected_action:
            raise ValueError("selected action does not bit-match its offered candidate")
        if is_official_fallback:
            if (
                official_action is None
                or official_action.get("action") != selected_action
            ):
                raise ValueError(
                    "official fallback action does not match the caller's official action"
                )
        execution_result = execute_mem_action(copy.deepcopy(selected_action))
        result = {
            "schema": "mem_psg_action_adapter_result_v1",
            "enabled": True,
            "arm_id": str(arm_id),
            "decision": decision,
            "executed_candidate_id": candidate_id,
            "execution_result": execution_result,
            "executes_action": True,
            "decision_provider_called": True,
            "official_policy_untouched": True,
            "config": asdict(self.config),
        }
        result["decision"]["executes_action"] = True
        json.dumps(result, allow_nan=False)
        return result


__all__ = [
    "MemActionExecutionState",
    "OfficialMemActionExecutor",
    "PsgMemActionAdapter",
    "PsgMemActionAdapterConfig",
    "build_frozen_mem_action_candidates",
    "score_official_mem_push_candidates",
]
