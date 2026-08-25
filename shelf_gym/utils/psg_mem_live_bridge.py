"""Default-off live bridge from frozen MEM decisions to the X3 callbacks.

The bridge owns no simulator logic and does not import ``scene_graph_mem``.
It calls the existing MEM view/push candidate and scoring methods, packages a
single frozen action set, asks injected sidecar callbacks to build/bind the
graph, and executes the selected token through :class:`OfficialMemActionExecutor`.
When disabled it invokes none of those callbacks and leaves ``mem.run`` as the
sole authority.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from .psg_mem_action_adapter import (
    MemActionExecutionState,
    OfficialMemActionExecutor,
    build_frozen_mem_action_candidates,
    score_official_mem_push_candidates,
)


_QUERY_FIELDS = {"class_id", "class_probs", "coarse_region"}


@dataclass(frozen=True)
class OfficialMemStepBridgeConfig:
    enabled: bool = False
    action_budget: int = 8
    use_push: bool = True
    first_push_step: int = 3
    treatment_first_push_step: int = 1
    reserve_final_observation: bool = True
    map_width: int = 200
    crop_row_offset: int = 10


def _coerce_config(
    value: OfficialMemStepBridgeConfig | Mapping[str, Any] | None,
) -> OfficialMemStepBridgeConfig:
    if value is None:
        result = OfficialMemStepBridgeConfig()
    elif isinstance(value, OfficialMemStepBridgeConfig):
        result = value
    elif isinstance(value, Mapping):
        result = OfficialMemStepBridgeConfig(**dict(value))
    else:
        raise TypeError("live bridge config must be a mapping or dataclass")
    for name in ("enabled", "use_push", "reserve_final_observation"):
        if not isinstance(getattr(result, name), bool):
            raise ValueError(f"live bridge {name} must be boolean")
    for name in ("action_budget", "map_width"):
        raw = getattr(result, name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"live bridge {name} must be a positive integer")
    for name in ("first_push_step", "treatment_first_push_step"):
        raw = getattr(result, name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"live bridge {name} must be non-negative")
    if (
        isinstance(result.crop_row_offset, bool)
        or not isinstance(result.crop_row_offset, int)
        or result.crop_row_offset < 0
    ):
        raise ValueError("live bridge crop_row_offset must be non-negative")
    return result


def _validate_query(query: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(query, Mapping):
        raise TypeError("planner query must be a mapping")
    unexpected = sorted(set(query) - _QUERY_FIELDS)
    if unexpected:
        raise ValueError(f"planner query contains privileged fields: {unexpected}")
    if "class_id" not in query and "class_probs" not in query:
        raise ValueError("planner query requires class_id or class_probs")
    result = copy.deepcopy(dict(query))
    json.dumps(result, allow_nan=False)
    return result


def _finite_ig_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array.copy()


def _typed_candidate_key(value: Any) -> tuple[type, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("candidate IDs must be strings or integers")
    return type(value), value


class OfficialMemStepBridge:
    """Expose one stateful official-MEM episode through X3 callback shapes."""

    def __init__(
        self,
        mem: Any,
        *,
        state: MemActionExecutionState,
        action_executor: OfficialMemActionExecutor,
        pipeline_module: Any,
        graph_provider: Callable[..., Mapping[str, Any]],
        candidate_binder: Callable[
            [Mapping[str, Any], Sequence[Mapping[str, Any]]],
            Sequence[Mapping[str, Any]],
        ],
        episode_id: str,
        target_query: Mapping[str, Any],
        push_score_provider: Callable[..., Mapping[str, Any]] | None = None,
        config: OfficialMemStepBridgeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(state, MemActionExecutionState):
            raise TypeError("state must be MemActionExecutionState")
        if not isinstance(action_executor, OfficialMemActionExecutor):
            raise TypeError("action_executor must be OfficialMemActionExecutor")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        for name, callback in (
            ("graph_provider", graph_provider),
            ("candidate_binder", candidate_binder),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if push_score_provider is not None and not callable(push_score_provider):
            raise TypeError("push_score_provider must be callable")
        self.mem = mem
        self.state = state
        self.action_executor = action_executor
        self.pipeline_module = pipeline_module
        self.graph_provider = graph_provider
        self.candidate_binder = candidate_binder
        self.episode_id = episode_id
        self.target_query = _validate_query(target_query)
        self.push_score_provider = (
            score_official_mem_push_candidates
            if push_score_provider is None
            else push_score_provider
        )
        self.config = _coerce_config(config)
        self.done_mapping = False
        self.pending_post_push_viewpoint: int | None = None
        self.history: list[Dict[str, Any]] = []

    def _select_viewpoint(self) -> tuple[int, float, str]:
        if self.state.fresh_push:
            if self.pending_post_push_viewpoint is None:
                raise RuntimeError("post-push observation has no frozen viewpoint")
            return self.pending_post_push_viewpoint, 0.0, "mandatory_post_push_view"
        method = getattr(self.pipeline_module, "get_igs_for_map", None)
        if not callable(method):
            raise TypeError("pipeline module does not expose get_igs_for_map")
        raw, _ = method(
            self.state.previous_map,
            self.mem.ig_calc,
            skip=1,
            use_alternative=True,
        )
        scores = _finite_ig_vector(raw, name="official view IG")
        for previous in self.state.previous_views:
            if (
                isinstance(previous, bool)
                or not isinstance(previous, (int, np.integer))
                or not 0 <= int(previous) < scores.size
            ):
                raise ValueError("previous MEM viewpoint is outside the camera array")
            scores[int(previous)] = 0.0
        return int(scores.argmax()), float(scores.max()), "official_view_ig"

    def _push_allowed(self, step: int) -> bool:
        upper = (
            int(self.config.action_budget) - 1
            if self.config.reserve_final_observation
            else int(self.config.action_budget)
        )
        return bool(
            self.config.use_push
            and int(step) >= int(self.config.first_push_step)
            and int(step) < upper
            and not self.done_mapping
            and not self.state.fresh_push
        )

    def _candidate_push_allowed(self, step: int, arm_id: str) -> bool:
        """Expose earlier pushes to retrieval arms without changing official MEM."""

        arm = str(arm_id).lower()
        if arm not in set("abcdefgh"):
            raise ValueError("live bridge arm_id must be a..h")
        if arm == "a":
            return self._push_allowed(step)
        upper = (
            int(self.config.action_budget) - 1
            if self.config.reserve_final_observation
            else int(self.config.action_budget)
        )
        return bool(
            self.config.use_push
            and int(step) >= int(self.config.treatment_first_push_step)
            and int(step) < upper
            and not self.done_mapping
            and not self.state.fresh_push
        )

    def step_provider(
        self,
        *,
        arm_id: str,
        episode_id: str,
        step: int,
        planner_query: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build the graph/action payload for one unexecuted budget step."""

        if not bool(self.config.enabled):
            return {
                "schema": "official_mem_live_step_bridge_v1",
                "enabled": False,
                "delegates_to_official_mem_run": True,
                "callbacks_called": False,
                "executes_action": False,
            }
        if episode_id != self.episode_id:
            raise ValueError("live bridge episode_id mismatch")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, np.integer))
            or int(step) != int(self.state.action_count)
        ):
            raise ValueError("live bridge step must equal completed budget steps")
        if int(step) >= int(self.config.action_budget):
            raise ValueError("live bridge action budget is exhausted")
        query = _validate_query(planner_query)
        if query != self.target_query:
            raise ValueError("live bridge planner query changed within the episode")
        if self.state.collision:
            raise RuntimeError("live MEM episode is terminal after object drop")

        viewpoint, one_step_view_ig, view_source = self._select_viewpoint()
        official_push_allowed = self._push_allowed(int(step))
        candidate_push_allowed = self._candidate_push_allowed(int(step), arm_id)
        push_data: Mapping[str, Any] = {"paths": None}
        score_record: Dict[str, Any] | None = None
        decision_view_ig = one_step_view_ig
        official_candidate_id = f"view:{viewpoint}"

        if candidate_push_allowed:
            subsequent_method = getattr(
                self.pipeline_module, "get_subsequent_igs_for_map", None
            )
            if not callable(subsequent_method):
                raise TypeError(
                    "pipeline module does not expose get_subsequent_igs_for_map"
                )
            subsequent = _finite_ig_vector(
                subsequent_method(
                    self.state.previous_map,
                    [viewpoint],
                    self.mem.ig_calc,
                ),
                name="official second-horizon view IG",
            )
            for previous in self.state.previous_views:
                if 0 <= int(previous) < subsequent.size:
                    subsequent[int(previous)] = 0.0
            decision_view_ig = float(one_step_view_ig + subsequent.max())
            candidate_method = getattr(self.mem, "get_possible_maps_push", None)
            if not callable(candidate_method):
                raise TypeError("MEM object does not expose get_possible_maps_push")
            candidate_kwargs: Dict[str, Any] = {"planner_camera_index": viewpoint}
            max_pushes = getattr(self.mem, "max_sampled_pushes", None)
            if max_pushes is not None:
                candidate_kwargs["num_points"] = int(max_pushes)
            push_data = candidate_method(
                self.state.previous_map,
                self.state.previous_semantic_map,
                **candidate_kwargs,
            )
            if not isinstance(push_data, Mapping):
                raise TypeError("MEM candidate generator must return a mapping")
            if push_data.get("paths") is not None:
                score_record = dict(
                    self.push_score_provider(
                        self.mem,
                        self.pipeline_module,
                        push_data,
                        previous_semantic_map=self.state.previous_semantic_map,
                        use_delta_H=True,
                        skip=5,
                    )
                )
                best_index = int(score_record["official_candidate_index"])
                best_push_ig = float(score_record["official_push_information_gain"])
                if not math.isfinite(best_push_ig):
                    raise ValueError("official push comparison IG must be finite")
                if official_push_allowed:
                    if best_push_ig > decision_view_ig:
                        official_candidate_id = f"push:{best_index}"
                    else:
                        # This no-op is the behavior of the frozen current main
                        # loop when a feasible push does not beat the two-view
                        # horizon.  It must not be silently rewritten as a view.
                        official_candidate_id = "noop:official_policy"
                self.pending_post_push_viewpoint = viewpoint
            elif official_push_allowed:
                official_candidate_id = "noop:official_policy"

        push_scores = (
            []
            if score_record is None
            else list(score_record["candidate_information_gains"])
        )
        unbound = build_frozen_mem_action_candidates(
            push_data,
            viewpoint=viewpoint,
            viewpoint_information_gain=decision_view_ig,
            push_information_gains=push_scores,
            map_width=int(self.config.map_width),
            crop_row_offset=int(self.config.crop_row_offset),
            include_noop=bool(official_push_allowed),
        )
        unbound[0]["one_step_information_gain"] = float(one_step_view_ig)
        unbound[0]["decision_horizon_information_gain"] = float(decision_view_ig)
        graph = self.graph_provider(
            episode_id=self.episode_id,
            step=int(step),
            occupancy_distribution=self.state.previous_map,
            semantic_concentration=self.state.previous_semantic_map,
            target_query=copy.deepcopy(self.target_query),
            selected_view_indices=list(self.state.previous_views),
            push_data=push_data,
            action_candidates=copy.deepcopy(unbound),
            previous_execution=(
                None if not self.history else copy.deepcopy(self.history[-1])
            ),
        )
        if not isinstance(graph, Mapping):
            raise TypeError("graph_provider must return a graph mapping")
        bound = self.candidate_binder(graph, unbound)
        if isinstance(bound, (str, bytes)) or not isinstance(bound, Sequence):
            raise TypeError("candidate_binder must return a candidate sequence")
        candidates = [copy.deepcopy(dict(row)) for row in bound]
        official_key = _typed_candidate_key(official_candidate_id)
        official = next(
            (
                row
                for row in candidates
                if _typed_candidate_key(row.get("candidate_id")) == official_key
            ),
            None,
        )
        if official is None:
            raise RuntimeError(
                "official decision is absent from the frozen candidate set"
            )
        self.action_executor.set_push_candidates(push_data)
        diagnostics = {
            "schema": "official_mem_live_step_diagnostics_v1",
            "arm_id": str(arm_id),
            "step": int(step),
            "viewpoint": viewpoint,
            "viewpoint_source": view_source,
            "push_allowed": official_push_allowed,
            "candidate_push_allowed": candidate_push_allowed,
            "treatment_early_push_opportunity": bool(
                candidate_push_allowed and not official_push_allowed
            ),
            "fresh_push": bool(self.state.fresh_push),
            "done_mapping": bool(self.done_mapping),
            "official_candidate_id": official_candidate_id,
            "official_push_scoring": copy.deepcopy(score_record),
            "official_main_noop_preserved": official_candidate_id
            == "noop:official_policy",
        }
        json.dumps(diagnostics, allow_nan=False)
        return {
            "graph": graph,
            "action_candidates": candidates,
            "official_action": {
                "candidate_id": official["candidate_id"],
                "source_node_id": official.get("source_node_id"),
                "action": copy.deepcopy(official["action"]),
            },
            "bridge_diagnostics": diagnostics,
        }

    def execute_mem_action(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute one selected token, then update frozen MEM stopping state."""

        if not bool(self.config.enabled):
            raise RuntimeError("disabled live bridge delegates to mem.run")
        kind = action.get("kind") if isinstance(action, Mapping) else None
        if kind == "push" and self.pending_post_push_viewpoint is None:
            raise RuntimeError("push action has no frozen post-push viewpoint")
        result = self.action_executor.execute(action)
        if kind in {"observe", "view", "noop"} or bool(result.get("collision", False)):
            self.pending_post_push_viewpoint = None
        mapped_fraction = None
        certainty_method = getattr(self.mem, "get_semantic_certainty", None)
        mapped_method = getattr(self.mem, "get_certainly_mapped_fraction", None)
        if callable(certainty_method) and callable(mapped_method):
            certainty = certainty_method(self.state.previous_semantic_map)
            mapped_fraction = float(
                mapped_method(certainty, float(self.mem.prob_cutoff))
            )
            if not math.isfinite(mapped_fraction):
                raise ValueError("MEM mapped fraction must be finite")
            self.done_mapping = bool(
                mapped_fraction >= float(self.mem.stopping_criterion)
            )
        completed = copy.deepcopy(dict(result))
        completed["selected_action"] = copy.deepcopy(dict(action))
        completed["mapped_fraction"] = mapped_fraction
        completed["done_mapping"] = bool(self.done_mapping)
        completed["official_main_untouched"] = True
        completed["bridge_config"] = asdict(self.config)
        self.history.append(copy.deepcopy(completed))
        json.dumps(completed, allow_nan=False)
        return completed


__all__ = ["OfficialMemStepBridge", "OfficialMemStepBridgeConfig"]
