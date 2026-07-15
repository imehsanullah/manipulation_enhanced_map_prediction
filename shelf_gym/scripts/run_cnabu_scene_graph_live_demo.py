#!/usr/bin/env python3
"""Live MEM/PyBullet CNABU scene-graph demo.

This script is a reusable MEM-side runtime entrypoint. It starts the MEM
PyBullet environment, performs one or more CNABU observation updates, generates
a scene graph from the live CNABU belief tensors after each update, and can
optionally update a map-anchored OpenCV graph visualization and save lightweight
diagnostics. The live default follows the validated primary method: frozen
learned component splitting with deterministic geometric relation edges.

It does not train, export datasets, write checkpoints, or use GT/simulator
instance labels as graph input.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from shelf_gym.scripts.run_cnabu_pipeline import ManipulationEnhancedMapping
from scene_graph_mem.runtime.cnabu_learned_component_splitter import (
    DEFAULT_CHECKPOINT_PATH,
    LearnedCnabuComponentSplitter,
)
from scene_graph_mem.runtime.cnabu_scene_graph import (
    DEFAULT_YCB_CLASS_NAMES,
    build_blocks_access_edges,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
    predict_scene_graph_from_cnabu,
)
from scene_graph_mem.runtime.cnabu_scene_graph_viz import (
    DEFAULT_CLASS_PALETTE_BGR,
    SceneGraphDisplayTracker,
    build_cnabu_map_context,
    render_cnabu_belief_map_view,
    render_cnabu_context_background,
    render_cnabu_scene_graph_research_view,
)
from scene_graph_mem.relations.path_aligned_features import (
    reconstruct_sparse_node_voxel_support,
)
from shelf_gym.utils.model_evaluation_utils import get_igs_for_map, get_subsequent_igs_for_map
from shelf_gym.utils.pushing_utils import execute_push
from shelf_gym.utils.action_conditioned_relation_oracle import (
    build_cnabu_runtime_candidate_action_mask,
)


THESIS_ROOT = Path("/home/user/ehsanullahm1/thesis")
DEFAULT_DIAGNOSTICS_PARENT = THESIS_ROOT / "thesis_records" / "diagnostics"
DEFAULT_RANKED_RELATION_CONFIG = (
    THESIS_ROOT
    / "scene_graph_mem/configs/cnabu/"
    "scene_graph_action_v1_candidate_planner_evidence_hybrid_planner_swept_path_resolved.yaml"
)
DEFAULT_RANKED_RELATION_CHECKPOINT = (
    THESIS_ROOT
    / "scene_graph_mem/checkpoints/"
    "cnabu_action_relation_v1_340_fresh_planner_swept_path_resolved_seed1_20260715/"
    "model_best_validation.pth"
)
DEFAULT_RANKED_RELATION_THRESHOLD = 0.9
DEFAULT_SCENE_GRAPH_PYTHON = Path(
    "/home/user/ehsanullahm1/miniconda3/envs/scene_graph_mem/bin/python"
)
DEFAULT_RANKED_RELATION_BRIDGE = (
    THESIS_ROOT / "scene_graph_mem/tools/serve_ranked_relation_advisory.py"
)
CNABU_CANDIDATE_TRAJECTORY_RANKED_RELATION_V2_SCHEMA = (
    "cnabu_candidate_trajectory_ranked_relation_v2"
)
RAW_SHAPE_HW = (140, 200)
CROP_ROWS = (10, 130)
# The physical shelf spans approximately x=20..180 and y=40..118 in the
# 5 mm MEM grid. Eight pixels of fixed safety context keep borders and pushed
# objects visible without showing the much larger robot workspace.
DEFAULT_SHELF_VIEW_XYXY = (12, 32, 188, 126)
# ``get_processed_array_and_gt_data`` returns this fixed crop from the raw
# 140x200 MEM grid. Re-embedding it makes prediction and GT directly aligned.
GT_RAW_CROP_XYXY = (21, 35, 179, 119)
MODE_CONFIGS = {
    "split_off": {"enabled": False},
    "split_on_2d_candidate": {"enabled": True, "method": "candidate_gated_2d_footprint"},
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def default_diagnostics_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_DIAGNOSTICS_PARENT / f"mem_cnabu_scene_graph_live_demo_{timestamp}"


def parse_viewpoints(value: Optional[str], *, updates: int) -> List[int]:
    if updates < 1:
        raise ValueError("--updates must be at least 1")
    if value:
        viewpoints = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not viewpoints:
            raise ValueError("--viewpoints did not contain any integer viewpoint ids")
        return [viewpoints[index % len(viewpoints)] for index in range(updates)]
    if updates == 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, 299, num=updates)]


def tensor_shape(value: Any) -> List[int]:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return [int(dim) for dim in np.asarray(value).shape]


def graph_counts(graph: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    return {
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "classes": sorted({str(node.get("class_name")) for node in nodes}),
        "json_safe": bool(json.loads(json.dumps(graph, default=json_default))),
        "uses_gt": bool(graph.get("metadata", {}).get("uses_gt", False)),
        "requires_gt": bool(graph.get("metadata", {}).get("requires_gt", False)),
        "uses_simulator_instance_labels": bool(
            graph.get("metadata", {}).get("uses_simulator_instance_labels", False)
        ),
    }


def live_inputs_from_belief(
    *,
    occupancy_distribution: Any,
    semantic_concentration: Any,
    update_index: int,
    selected_view_indices: Sequence[int],
    viewpoint: int,
    update_kind: str = "observe",
) -> Dict[str, Any]:
    source = (
        "live_mem_pybullet_push_prediction"
        if str(update_kind) == "push_predicted"
        else "live_mem_pybullet_cnabu_update"
    )
    return {
        "occupancy_distribution": occupancy_distribution,
        "semantic_concentration": semantic_concentration,
        "raw_shape_hw": RAW_SHAPE_HW,
        "crop_rows": CROP_ROWS,
        "selected_view_indices": list(map(int, selected_view_indices)),
        "sample_id": f"live_mem_pybullet_update_{int(update_index):03d}_{update_kind}",
        "metadata": {
            "source": source,
            "gt_loaded": False,
            "simulator_instance_labels_used_for_graph_input": False,
            "viewpoint": int(viewpoint),
            "update_index": int(update_index),
            "update_kind": str(update_kind),
        },
    }


def run_graph_mode(
    *,
    mode: str,
    inputs: Mapping[str, Any],
    splitter: Optional[LearnedCnabuComponentSplitter],
) -> tuple[Dict[str, Any], Dict[str, float]]:
    started = time.perf_counter()
    if mode in MODE_CONFIGS:
        graph = predict_scene_graph_from_cnabu(
            **inputs,
            component_split_config=MODE_CONFIGS[mode],
            edge_config={"opening_side": "low"},
            include_masks=True,
        )
    elif mode == "learned_component_splitter":
        if splitter is None:
            raise ValueError("learned_component_splitter mode requires a loaded splitter")
        graph = splitter.predict_scene_graph(
            **inputs,
            edge_config={"opening_side": "low"},
            include_masks=True,
        )
    else:
        raise ValueError(f"unsupported scene graph mode: {mode}")

    total_seconds = float(time.perf_counter() - started)
    metadata = graph.setdefault("metadata", {})
    metadata["runtime_mode"] = mode
    metadata["update_index"] = int(inputs["metadata"]["update_index"])
    metadata["viewpoint"] = int(inputs["metadata"]["viewpoint"])
    timing = dict(metadata.get("runtime_timing_seconds", {}))
    timing["total_graph_generation"] = total_seconds
    metadata["runtime_timing_seconds"] = timing
    return graph, timing


def render_graph_image(
    *,
    graph: Mapping[str, Any],
    inputs: Mapping[str, Any],
    update_index: int,
    display_state: Mapping[str, Any],
    full_workspace_view: bool,
) -> np.ndarray:
    context = build_cnabu_map_context(
        occupancy_distribution=inputs.get("occupancy_distribution"),
        semantic_concentration=inputs.get("semantic_concentration"),
        raw_shape_hw=inputs.get("raw_shape_hw"),
        crop_rows=inputs.get("crop_rows"),
    )
    return render_cnabu_scene_graph_research_view(
        graph,
        context=context,
        update_index=int(update_index),
        width=1280,
        height=760,
        max_edges=36,
        display_state=display_state,
        view_xyxy=None if bool(full_workspace_view) else DEFAULT_SHELF_VIEW_XYXY,
        rotate_map_180=True,
    )


def render_belief_image(
    *,
    inputs: Mapping[str, Any],
    update_index: int,
    full_workspace_view: bool,
) -> np.ndarray:
    context = build_cnabu_map_context(
        occupancy_distribution=inputs.get("occupancy_distribution"),
        semantic_concentration=inputs.get("semantic_concentration"),
        raw_shape_hw=inputs.get("raw_shape_hw"),
        crop_rows=inputs.get("crop_rows"),
    )
    return render_cnabu_belief_map_view(
        context=context,
        update_index=int(update_index),
        width=640,
        height=520,
        title=(
            "Full-workspace CNABU/MEM belief"
            if bool(full_workspace_view)
            else "Shelf-focused CNABU/MEM belief"
        ),
        view_xyxy=None if bool(full_workspace_view) else DEFAULT_SHELF_VIEW_XYXY,
        rotate_map_180=True,
    )


def show_graph_window(
    *,
    image: np.ndarray,
    window_name: str,
    first_frame: bool,
) -> bool:
    try:
        if first_frame:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            image_h, image_w = image.shape[:2]
            window_scale = min(1680.0 / max(image_w, 1), 760.0 / max(image_h, 1))
            window_w = max(1, int(round(image_w * window_scale)))
            window_h = max(1, int(round(image_h * window_scale)))
            cv2.resizeWindow(window_name, window_w, window_h)
            cv2.moveWindow(window_name, 80 if window_w > 1280 else 600, 120)
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        return True
    except cv2.error as exc:
        print(f"Could not update graph window: {exc}", file=sys.stderr)
        return False


def _to_numpy_cpu(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "get") and callable(value.get):
        return np.asarray(value.get())
    return np.asarray(value)


def cnabu_mean_arrays_from_live_belief(
    occupancy_distribution: Any,
    semantic_concentration: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the live interleaved CNABU beliefs into finite mean arrays."""

    occupancy = _to_numpy_cpu(occupancy_distribution)
    semantic = _to_numpy_cpu(semantic_concentration)
    while occupancy.ndim > 3 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    while semantic.ndim > 3 and semantic.shape[0] == 1:
        semantic = semantic[0]
    if occupancy.ndim != 3 or occupancy.shape[0] % 2 != 0:
        raise ValueError(
            "live occupancy distribution must have interleaved beta/alpha shape [2Z,H,W]"
        )
    if semantic.ndim != 3 or semantic.shape[1:] != occupancy.shape[1:]:
        raise ValueError(
            "live semantic concentration must have shape [K,H,W] aligned with occupancy"
        )
    occupancy = np.asarray(occupancy, dtype=np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    if (
        not np.isfinite(occupancy).all()
        or not np.isfinite(semantic).all()
        or np.any(occupancy < 0.0)
        or np.any(semantic < 0.0)
    ):
        raise ValueError("live CNABU belief parameters must be finite and non-negative")
    beta = occupancy[0::2]
    alpha = occupancy[1::2]
    occupancy_mean = alpha / np.maximum(alpha + beta, 1.0e-8)
    semantic_mean = semantic / np.maximum(semantic.sum(axis=0, keepdims=True), 1.0e-8)
    return (
        occupancy_mean.astype(np.float32, copy=False),
        semantic_mean.astype(np.float32, copy=False),
    )


@dataclass
class RankedRelationAdvisoryRuntime:
    """Persistent cross-environment relation bridge; never an action executor."""

    config_path: Path
    checkpoint_path: Path
    threshold: float
    top_k: int
    target_node_id: Optional[int]
    device: str
    python_executable: Path = DEFAULT_SCENE_GRAPH_PYTHON
    bridge_script: Path = DEFAULT_RANKED_RELATION_BRIDGE
    environment: Any = None
    process: Optional[subprocess.Popen[str]] = None
    checkpoint_load: Optional[Mapping[str, Any]] = None
    bridge_resource_after_load: Optional[Mapping[str, Any]] = None
    load_seconds: float = 0.0
    request_count: int = 0

    def _read_protocol_message(self, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("ranked relation bridge is not running")
        while True:
            line = self.process.stdout.readline()
            if line == "":
                code = self.process.poll()
                raise RuntimeError(
                    "ranked relation bridge closed unexpectedly with code {}".format(code)
                )
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print("ranked relation bridge: {}".format(line.rstrip()), file=sys.stderr)
                continue
            if message.get("protocol") != "cnabu_ranked_relation_bridge_v1":
                continue
            if request_id is not None and message.get("request_id") != request_id:
                continue
            return dict(message)

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("ranked relation bridge is already started")
        for path, label in (
            (self.python_executable, "scene_graph_mem Python"),
            (self.bridge_script, "ranked relation bridge"),
            (self.config_path, "ranked relation config"),
            (self.checkpoint_path, "ranked relation checkpoint"),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError("missing {}: {}".format(label, path))
        command = [
            str(self.python_executable),
            str(self.bridge_script),
            "--config-file",
            str(self.config_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--device",
            str(self.device),
            "--threshold",
            str(float(self.threshold)),
        ]
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(THESIS_ROOT / "scene_graph_mem"),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        ready = self._read_protocol_message()
        if ready.get("status") != "ready":
            raise RuntimeError(
                "ranked relation bridge failed to start: {}".format(ready.get("error"))
            )
        self.load_seconds = float(time.perf_counter() - started)
        self.checkpoint_load = dict(ready["checkpoint_load"])
        self.bridge_resource_after_load = dict(ready["resource"])

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps(
                        {
                            "protocol": "cnabu_ranked_relation_bridge_v1",
                            "command": "shutdown",
                        }
                    )
                    + "\n"
                )
                process.stdin.flush()
                process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=10)

    def predict(
        self,
        *,
        graph: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if self.environment is None:
            raise RuntimeError("ranked advisory environment is not initialized")
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("ranked relation bridge is not started")
        started = time.perf_counter()
        graph_nodes = list(graph.get("nodes") or [])
        if len(graph_nodes) < 2:
            return {
                "schema": "mem_cnabu_ranked_relation_advisory_v1",
                "available": False,
                "unavailable_reason": "fewer_than_two_runtime_nodes",
                "executes_action": False,
                "timing_seconds": float(time.perf_counter() - started),
            }
        occupancy_mean, semantic_mean = cnabu_mean_arrays_from_live_belief(
            inputs["occupancy_distribution"],
            inputs["semantic_concentration"],
        )
        crop_rows = tuple(int(value) for value in inputs["crop_rows"])
        node_ids = tuple(
            int(node.get("component_id", node.get("id"))) for node in graph_nodes
        )
        node_classes = tuple(int(node["class_id"]) for node in graph_nodes)
        node_masks = np.stack(
            [decode_binary_mask_rle(node["mask"]) for node in graph_nodes], axis=0
        )
        support = reconstruct_sparse_node_voxel_support(
            occupancy_mean,
            semantic_mean,
            node_masks,
            node_classes,
            crop_rows=crop_rows,
        )
        action_mask = build_cnabu_runtime_candidate_action_mask(
            self.environment,
            self.environment.smg.hg,
            support.indices_zyx,
            crop_rows=crop_rows,
            node_ids=node_ids,
            initial_arm_config=np.asarray(
                self.environment.get_current_joint_config(), dtype=np.float64
            ),
            support_boundary_quantile=0.05,
            include_planner_swept_features=True,
        )
        self.request_count += 1
        request_id = "live_ranked_{:06d}".format(self.request_count)
        with tempfile.TemporaryDirectory(prefix="cnabu_ranked_relation_") as tmp:
            temporary_dir = Path(tmp)
            input_path = temporary_dir / "runtime_inputs.npz"
            graph_path = temporary_dir / "runtime_graph.json"
            action_mask_path = temporary_dir / "candidate_action_mask.json"
            output_path = temporary_dir / "ranked_advisory.json"
            np.savez_compressed(
                input_path,
                occupancy_mean=occupancy_mean,
                semantic_mean=semantic_mean,
                crop_rows=np.asarray(crop_rows, dtype=np.int64),
                raw_shape_hw=np.asarray(inputs["raw_shape_hw"], dtype=np.int64),
            )
            write_json(graph_path, graph)
            write_json(action_mask_path, action_mask)
            request = {
                "protocol": "cnabu_ranked_relation_bridge_v1",
                "command": "predict",
                "request_id": request_id,
                "input_npz": str(input_path),
                "runtime_graph_json": str(graph_path),
                "candidate_action_mask_json": str(action_mask_path),
                "output_json": str(output_path),
                "image_id": str(inputs["sample_id"]),
                "target_node_id": self.target_node_id,
                "top_k": int(self.top_k),
            }
            self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self.process.stdin.flush()
            response = self._read_protocol_message(request_id=request_id)
            if response.get("status") != "complete":
                raise RuntimeError(
                    "ranked relation bridge prediction failed: {}".format(
                        response.get("error")
                    )
                )
            advisory = json.loads(output_path.read_text(encoding="utf-8"))
        advisory["timing_seconds"] = float(time.perf_counter() - started)
        advisory["bridge_inference_seconds"] = response.get("elapsed_seconds")
        advisory["bridge_resource_after_inference"] = dict(
            response.get("resource") or {}
        )
        advisory["checkpoint_load"] = dict(self.checkpoint_load or {})
        advisory["cross_environment_bridge"] = {
            "runtime_python": sys.executable,
            "relation_python": str(self.python_executable),
            "reason": "manipulation_map does not provide detectron2; model inference stays in scene_graph_mem",
        }
        return advisory


def _align_gt_to_raw_view(
    semantic_gt: np.ndarray,
    occupancy_projection: np.ndarray,
    *,
    view_xyxy: Optional[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-embed the pre-cropped simulator GT in raw MEM coordinates."""

    semantic_gt = np.asarray(semantic_gt, dtype=np.int32)
    occupancy_projection = np.asarray(occupancy_projection, dtype=np.float32)
    if semantic_gt.ndim != 2 or occupancy_projection.ndim != 2:
        raise ValueError("semantic_gt and occupancy projection must be two-dimensional")

    gt_x1, gt_y1, gt_x2, gt_y2 = GT_RAW_CROP_XYXY
    gt_width = int(gt_x2 - gt_x1)
    gt_height = int(gt_y2 - gt_y1)
    if semantic_gt.shape != (gt_height, gt_width):
        semantic_gt = cv2.resize(
            semantic_gt.astype(np.float32),
            (gt_width, gt_height),
            interpolation=cv2.INTER_NEAREST,
        ).round().astype(np.int32)
    if occupancy_projection.shape != (gt_height, gt_width):
        occupancy_projection = cv2.resize(
            occupancy_projection,
            (gt_width, gt_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

    raw_height, raw_width = RAW_SHAPE_HW
    raw_semantic = np.full((raw_height, raw_width), len(DEFAULT_CLASS_PALETTE_BGR) - 1, dtype=np.int32)
    raw_occupancy = np.zeros((raw_height, raw_width), dtype=np.float32)
    raw_valid = np.zeros((raw_height, raw_width), dtype=bool)
    raw_semantic[gt_y1:gt_y2, gt_x1:gt_x2] = semantic_gt
    raw_occupancy[gt_y1:gt_y2, gt_x1:gt_x2] = occupancy_projection
    raw_valid[gt_y1:gt_y2, gt_x1:gt_x2] = True

    if view_xyxy is None:
        view_x1, view_y1, view_x2, view_y2 = 0, 0, raw_width, raw_height
    else:
        coordinates = np.asarray(view_xyxy).reshape(-1)
        if coordinates.size != 4:
            raise ValueError(f"view_xyxy must contain four entries, got {view_xyxy!r}")
        view_x1, view_y1, view_x2, view_y2 = [int(value) for value in coordinates.tolist()]
        if (
            view_x1 < 0
            or view_y1 < 0
            or view_x2 > raw_width
            or view_y2 > raw_height
            or view_x2 <= view_x1
            or view_y2 <= view_y1
        ):
            raise ValueError(f"view_xyxy lies outside the raw MEM grid: {view_xyxy!r}")

    view_slice = np.s_[view_y1:view_y2, view_x1:view_x2]
    return raw_semantic[view_slice], raw_occupancy[view_slice], raw_valid[view_slice]


def _gt_instance_stack_in_raw_frame(value: Any) -> np.ndarray:
    stack = np.asarray(_to_numpy_cpu(value))
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"GT instance_maps must have shape [H,W] or [V,H,W], got {stack.shape}")
    if tuple(stack.shape[-2:]) == RAW_SHAPE_HW:
        return stack

    gt_x1, gt_y1, gt_x2, gt_y2 = GT_RAW_CROP_XYXY
    gt_shape = (gt_y2 - gt_y1, gt_x2 - gt_x1)
    if tuple(stack.shape[-2:]) == gt_shape:
        raw = np.zeros((stack.shape[0], *RAW_SHAPE_HW), dtype=stack.dtype)
        raw[:, gt_y1:gt_y2, gt_x1:gt_x2] = stack
        return raw
    raise ValueError(
        f"GT instance-map spatial shape {stack.shape[-2:]} is neither raw {RAW_SHAPE_HW} nor crop {gt_shape}"
    )


def _majority_gt_class(semantic_gt_raw: np.ndarray, mask: np.ndarray) -> Optional[int]:
    values = np.asarray(semantic_gt_raw)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    labels = np.rint(values).astype(np.int64)
    labels = labels[(labels >= 0) & (labels < len(DEFAULT_YCB_CLASS_NAMES) - 1)]
    if labels.size == 0:
        return None
    unique, counts = np.unique(labels, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def build_gt_instance_scene_graph(
    gt_data: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build an evaluation-only graph from privileged simulator instance masks."""

    if "instance_maps" not in gt_data:
        raise KeyError("GT comparison requires simulator instance_maps")

    semantic_crop = np.asarray(_to_numpy_cpu(gt_data["semantic_gt"]), dtype=np.int32)
    occupancy_gt = np.asarray(_to_numpy_cpu(gt_data["voxel_height_map"]), dtype=np.float32)
    occupancy_projection = occupancy_gt.max(axis=2) if occupancy_gt.ndim == 3 else occupancy_gt
    aligned_semantic, raw_occupancy, _ = _align_gt_to_raw_view(
        semantic_crop,
        occupancy_projection,
        view_xyxy=None,
    )
    semantic_gt_raw = np.asarray(
        _to_numpy_cpu(gt_data.get("semantic_gt_raw", aligned_semantic)),
        dtype=np.int32,
    )
    if semantic_gt_raw.shape != RAW_SHAPE_HW:
        raise ValueError(
            f"semantic_gt_raw shape {semantic_gt_raw.shape} must match raw MEM shape {RAW_SHAPE_HW}"
        )

    instance_stack = _gt_instance_stack_in_raw_frame(gt_data["instance_maps"])
    finite = instance_stack[np.isfinite(instance_stack)]
    map_instance_ids = {
        int(round(float(value)))
        for value in np.unique(finite)
        if float(value) > 0.0 and np.isclose(value, round(float(value)))
    }
    declared_instance_ids = np.asarray(gt_data.get("object_instance_ids", []), dtype=np.int64).reshape(-1)
    declared_class_ids = np.asarray(gt_data.get("object_class_ids", []), dtype=np.int64).reshape(-1)
    if declared_instance_ids.size != declared_class_ids.size:
        raise ValueError("object_instance_ids and object_class_ids must have the same length")
    declared_classes = {
        int(instance_id): int(class_id)
        for instance_id, class_id in zip(declared_instance_ids.tolist(), declared_class_ids.tolist())
    }
    candidate_ids = sorted(map_instance_ids & set(declared_classes)) if declared_classes else sorted(map_instance_ids)

    nodes: List[Dict[str, Any]] = []
    for simulator_instance_id in candidate_ids:
        mask = np.any(instance_stack == int(simulator_instance_id), axis=0)
        class_id = declared_classes.get(
            int(simulator_instance_id),
            _majority_gt_class(semantic_gt_raw, mask),
        )
        if class_id is None or not bool(mask.any()):
            continue
        ys, xs = np.nonzero(mask)
        node_id = len(nodes) + 1
        nodes.append(
            {
                "id": int(node_id),
                "simulator_instance_id": int(simulator_instance_id),
                "class_id": int(class_id),
                "class_name": str(DEFAULT_YCB_CLASS_NAMES[int(class_id)]),
                "bbox_xyxy_abs": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "centroid_yx": [float(ys.mean()), float(xs.mean())],
                "area_pixels": int(mask.sum()),
                "score": 1.0,
                "mask": encode_binary_mask_rle(mask),
                "source": "simulator_gt_instance_masks",
            }
        )

    edge_config = {
        "relation": "blocks_access_to",
        "access_axis": "y",
        "opening_side": "low",
        "min_front_gap": 0.0,
        "min_lateral_overlap": 0.0,
        "lateral_overlap_mode": "union",
    }
    edges, edge_metadata = build_blocks_access_edges(
        nodes,
        config=edge_config,
        image_shape_hw=RAW_SHAPE_HW,
    )
    adjacency = [[0 for _ in nodes] for _ in nodes]
    for edge in edges:
        adjacency[int(edge["source_index"])][int(edge["target_index"])] = 1

    graph: Dict[str, Any] = {
        "schema": "mem_gt_instance_scene_graph_v0",
        "nodes": nodes,
        "edges": edges,
        "adjacency_matrix": adjacency,
        "thresholds": {"edge_rule": edge_config},
        "metadata": {
            "node_source": "simulator_gt_instance_masks",
            "runtime_mode": "simulator_gt_instance_masks",
            "num_nodes": int(len(nodes)),
            "num_edges": int(len(edges)),
            "raw_shape_hw": list(RAW_SHAPE_HW),
            "instance_map_views": int(instance_stack.shape[0]),
            "uses_gt": True,
            "requires_gt": True,
            "uses_simulator_instance_labels": True,
            "edge_metadata": edge_metadata,
            "evaluation_only": True,
        },
    }
    background = render_cnabu_context_background(
        occupancy_projection=raw_occupancy,
        semantic_labels=semantic_gt_raw,
        semantic_confidence=np.ones(RAW_SHAPE_HW, dtype=np.float32),
    )
    context: Dict[str, Any] = {
        "background_bgr": background,
        "raw_shape_hw": list(RAW_SHAPE_HW),
    }
    return graph, context


def render_gt_topdown_panel(
    gt_data: Mapping[str, Any],
    *,
    width: int = 1280,
    height: int = 760,
    update_index: Optional[int] = None,
    view_xyxy: Optional[Sequence[int]] = DEFAULT_SHELF_VIEW_XYXY,
    rotate_180: bool = True,
) -> np.ndarray:
    """Render an evaluation-only GT instance graph in the MEM map frame."""

    graph, context = build_gt_instance_scene_graph(gt_data)
    return render_cnabu_scene_graph_research_view(
        graph,
        context=context,
        update_index=update_index,
        width=int(width),
        height=int(height),
        max_edges=36,
        view_xyxy=view_xyxy,
        rotate_map_180=bool(rotate_180),
        title="Ground-truth Scene Graph",
        subtitle="Simulator instance masks (evaluation only; not used by CNABU inference)",
        method_label="independent GT node IDs; deterministic rule edges",
    )


def compose_graph_gt_panel(graph_bgr: np.ndarray, gt_bgr: np.ndarray) -> np.ndarray:
    return compose_diagnostic_panels([graph_bgr, gt_bgr])


def compose_diagnostic_panels(panels: Sequence[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("at least one panel is required")
    if len(panels) == 1:
        return np.asarray(panels[0])[:, :, :3].copy()
    height = int(max(np.asarray(panel).shape[0] for panel in panels))
    gap = 16
    fitted_panels = []
    for panel in panels:
        panel = np.asarray(panel)[:, :, :3]
        panel_w = int(round(height * panel.shape[1] / max(panel.shape[0], 1)))
        fitted_panels.append(cv2.resize(panel, (panel_w, height), interpolation=cv2.INTER_AREA))
    width = int(sum(panel.shape[1] for panel in fitted_panels) + gap * (len(fitted_panels) - 1))
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    x = 0
    for index, panel in enumerate(fitted_panels):
        canvas[:, x:x + panel.shape[1]] = panel
        x += panel.shape[1]
        if index < len(fitted_panels) - 1:
            cv2.rectangle(canvas, (x, 0), (x + gap, height), (230, 232, 234), -1)
            x += gap
    return canvas


def get_gt_diagnostic_data(
    *,
    mem: ManipulationEnhancedMapping,
    args: argparse.Namespace,
    summary: Dict[str, Any],
) -> Optional[Mapping[str, Any]]:
    if not bool(args.show_gt_panel):
        return None
    _, gt_data = mem.get_processed_array_and_gt_data(only_gt=True)
    summary["gt_diagnostic_loaded"] = True
    return gt_data


def camera_array_target_for_viewpoint(viewpoint: int) -> np.ndarray:
    index = int(viewpoint)
    if index < 100:
        return np.asarray([-0.3, 0.95, 1.07], dtype=np.float64)
    if index < 200:
        return np.asarray([0.3, 0.95, 1.07], dtype=np.float64)
    return np.asarray([0.0, 0.95, 1.07], dtype=np.float64)


def klampt_config_from_pybullet_joints(
    mem: ManipulationEnhancedMapping,
    joint_positions: Sequence[float],
) -> List[float]:
    current_config = copy.deepcopy(mem.klampt_utils.robot.getConfig())
    mem.klampt_utils.set_joint_positions(list(joint_positions))
    config = copy.deepcopy(mem.klampt_utils.robot.getConfig())
    mem.klampt_utils.robot.setConfig(current_config)
    return config


def execute_klampt_path(
    *,
    mem: ManipulationEnhancedMapping,
    path: Sequence[Sequence[float]],
    annotation: str = "free",
    verbose: bool = False,
) -> int:
    annotations = [str(annotation)] * len(path)
    moved_positions = mem.linear_interpolate_motion_klampt_joint_traj(
        list(path),
        traj_annotation=annotations,
        imagined=False,
        verbose=verbose,
    )
    return int(len(moved_positions))


def move_arm_home_for_observation(
    *,
    mem: ManipulationEnhancedMapping,
    just_endpoints: bool,
    verbose: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    start_config = klampt_config_from_pybullet_joints(mem, mem.get_current_arm_and_gripper_joint_config())
    home_joints = list(getattr(mem, "init_arm_and_gripper_joint_config", mem.get_current_arm_and_gripper_joint_config()))
    home_joints[:6] = list(mem.initial_parameters[:6])
    if len(home_joints) > 6:
        home_joints[6] = float(mem.initial_parameters[6])
    home_config = klampt_config_from_pybullet_joints(mem, home_joints)
    current_config = copy.deepcopy(mem.klampt_utils.robot.getConfig())
    mem.klampt_utils.robot.setConfig(start_config)
    path = mem.klampt_utils.plan_to_joint_goal(
        home_config,
        check_feasibility=False,
        verbose=verbose,
        just_endpoints=just_endpoints,
    )
    mem.klampt_utils.robot.setConfig(current_config)
    if path is None:
        mem.reset_robot(mem.initial_parameters)
        return {
            "home_return_success": False,
            "home_return_fallback_reset": True,
            "home_return_waypoints": 0,
            "home_return_motion_samples": 0,
            "home_return_seconds": float(time.perf_counter() - started),
        }
    samples = execute_klampt_path(mem=mem, path=path, annotation="free", verbose=verbose)
    return {
        "home_return_success": True,
        "home_return_fallback_reset": False,
        "home_return_waypoints": int(len(path)),
        "home_return_motion_samples": int(samples),
        "home_return_seconds": float(time.perf_counter() - started),
    }


def move_arm_for_observation(
    *,
    mem: ManipulationEnhancedMapping,
    viewpoint: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "arm_observe_motion_enabled": bool(args.move_arm_for_observe),
        "arm_observe_motion_success": False,
        "arm_observe_motion_skipped": True,
        "arm_observe_motion_target_link": str(args.observe_arm_target_link),
        "arm_observe_motion_sensor_source": "camera_array",
        "arm_observe_motion_note": (
            "UR5 camera link is moved to the matching camera-array pose for visualization; "
            "the CNABU observation still uses MEM's camera-array sensor pipeline."
        ),
    }
    if not bool(args.move_arm_for_observe):
        return record

    started = time.perf_counter()
    index = int(viewpoint)
    if index < 0 or index >= len(mem.camera_array_cams):
        record.update(
            {
                "arm_observe_motion_error": f"viewpoint {index} outside camera array range",
                "arm_observe_motion_seconds": float(time.perf_counter() - started),
            }
        )
        return record

    camera_info = mem.camera_array_cams[index]
    target_position = np.asarray(camera_info["position"], dtype=np.float64)
    target_point = camera_array_target_for_viewpoint(index)
    try:
        target_link = mem.klampt_utils.robot.link(str(args.observe_arm_target_link))
    except Exception as exc:  # noqa: BLE001 - Klampt may raise non-standard exceptions.
        record.update(
            {
                "arm_observe_motion_error": f"missing target link: {exc}",
                "arm_observe_motion_seconds": float(time.perf_counter() - started),
            }
        )
        return record

    start_config = mem.get_current_arm_and_gripper_joint_config()
    target_joint_config, path, _ = mem.klampt_utils.test_feasibility(
        start_config=start_config,
        target_pose=target_position,
        target_direction=target_point,
        target_link=target_link,
        is_pybullet_config=True,
        just_endpoints=not bool(args.observe_arm_full_plan),
        verbose=bool(args.observe_arm_verbose),
    )
    if path is None or target_joint_config is None:
        record.update(
            {
                "arm_observe_motion_error": "IK or path planning failed",
                "arm_observe_motion_target_position": target_position.tolist(),
                "arm_observe_motion_target_point": target_point.tolist(),
                "arm_observe_motion_seconds": float(time.perf_counter() - started),
            }
        )
        return record

    samples = execute_klampt_path(
        mem=mem,
        path=path,
        annotation="free",
        verbose=bool(args.observe_arm_verbose),
    )
    record.update(
        {
            "arm_observe_motion_success": True,
            "arm_observe_motion_skipped": False,
            "arm_observe_motion_waypoints": int(len(path)),
            "arm_observe_motion_samples": int(samples),
            "arm_observe_motion_target_position": target_position.tolist(),
            "arm_observe_motion_target_point": target_point.tolist(),
            "arm_observe_motion_seconds": float(time.perf_counter() - started),
        }
    )
    if float(args.observe_arm_pause_sec) > 0.0:
        time.sleep(float(args.observe_arm_pause_sec))
        record["arm_observe_motion_pause_seconds"] = float(args.observe_arm_pause_sec)
    if not bool(args.observe_arm_keep_pose):
        record.update(
            move_arm_home_for_observation(
                mem=mem,
                just_endpoints=not bool(args.observe_arm_full_plan),
                verbose=bool(args.observe_arm_verbose),
            )
        )
    return record


def wait_after_update(*, sleep_sec: float, graph_window_open: bool) -> None:
    if float(sleep_sec) <= 0.0:
        return
    if graph_window_open:
        cv2.waitKey(max(1, int(float(sleep_sec) * 1000)))
    else:
        time.sleep(float(sleep_sec))


def process_graph_update(
    *,
    args: argparse.Namespace,
    summary: Dict[str, Any],
    diagnostics_dir: Path,
    mode: str,
    splitter: Optional[LearnedCnabuComponentSplitter],
    display_tracker: SceneGraphDisplayTracker,
    occupancy_distribution: Any,
    semantic_concentration: Any,
    update_index: int,
    update_kind: str,
    viewpoint: int,
    selected_view_indices: Sequence[int],
    cnabu_update_seconds: Optional[float],
    graph_window_open: bool,
    window_name: str,
    gt_data: Optional[Mapping[str, Any]] = None,
    extra_record: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Any], bool]:
    inputs = live_inputs_from_belief(
        occupancy_distribution=occupancy_distribution,
        semantic_concentration=semantic_concentration,
        update_index=update_index,
        selected_view_indices=selected_view_indices,
        viewpoint=int(viewpoint),
        update_kind=update_kind,
    )
    if not summary["live_tensor_shapes"]:
        summary["live_tensor_shapes"] = {
            "occupancy_distribution": tensor_shape(occupancy_distribution),
            "semantic_concentration": tensor_shape(semantic_concentration),
            "raw_shape_hw": list(RAW_SHAPE_HW),
            "crop_rows": list(CROP_ROWS),
        }

    graph, graph_timing = run_graph_mode(
        mode=mode,
        inputs=inputs,
        splitter=splitter,
    )
    graph["metadata"]["update_kind"] = str(update_kind)
    ranked_advisory: Optional[Dict[str, Any]] = None
    ranked_runtime = getattr(args, "_ranked_relation_runtime", None)
    if ranked_runtime is not None:
        ranked_advisory = ranked_runtime.predict(graph=graph, inputs=inputs)
        graph["ranked_relation_advisory"] = ranked_advisory
        graph["metadata"].update(
            {
                "ranked_relation_advisory_enabled": True,
                "ranked_relation_advisory_executes_action": False,
                "ranked_relation_schema": (
                    ranked_advisory.get("canonical_ranked_relation", {}).get("schema")
                    if ranked_advisory.get("available")
                    else None
                ),
            }
        )
        graph_timing["ranked_relation_advisory"] = float(
            ranked_advisory["timing_seconds"]
        )
    counts = graph_counts(graph)
    if counts["uses_gt"] or counts["requires_gt"] or counts["uses_simulator_instance_labels"]:
        raise RuntimeError(f"graph input safety violation at update {update_index}: {counts}")
    display_state = display_tracker.update(graph)

    image: Optional[np.ndarray] = None
    belief_image: Optional[np.ndarray] = None
    json_path: Optional[Path] = None
    png_path: Optional[Path] = None
    belief_png_path: Optional[Path] = None
    if bool(args.show_graph) or bool(args.save_diagnostics):
        graph_image = render_graph_image(
            graph=graph,
            inputs=inputs,
            update_index=update_index,
            display_state=display_state,
            full_workspace_view=bool(args.full_workspace_view),
        )
        panels = [graph_image]
        gt_panel: Optional[np.ndarray] = None
        if bool(args.show_belief_panel):
            belief_image = render_belief_image(
                inputs=inputs,
                update_index=update_index,
                full_workspace_view=bool(args.full_workspace_view),
            )
            panels = [belief_image, graph_image]
        if bool(args.show_gt_panel) and gt_data is not None:
            gt_panel = render_gt_topdown_panel(
                gt_data,
                width=graph_image.shape[1],
                height=graph_image.shape[0],
                update_index=update_index,
                view_xyxy=(
                    None if bool(args.full_workspace_view) else DEFAULT_SHELF_VIEW_XYXY
                ),
                rotate_180=True,
            )
            panels.append(gt_panel)
        image = (
            compose_graph_gt_panel(graph_image, gt_panel)
            if gt_panel is not None and not bool(args.show_belief_panel)
            else compose_diagnostic_panels(panels)
        )
    if bool(args.save_diagnostics):
        safe_kind = str(update_kind).replace(" ", "_")
        json_path = diagnostics_dir / f"update_{update_index:03d}_{safe_kind}_graph.json"
        png_path = diagnostics_dir / f"update_{update_index:03d}_{safe_kind}_graph.png"
        write_json(json_path, graph)
        if image is not None:
            cv2.imwrite(str(png_path), image)
        if belief_image is not None:
            belief_png_path = diagnostics_dir / f"update_{update_index:03d}_{safe_kind}_belief.png"
            cv2.imwrite(str(belief_png_path), belief_image)
    if bool(args.show_graph) and image is not None:
        graph_window_open = show_graph_window(
            image=image,
            window_name=window_name,
            first_frame=not graph_window_open,
        )

    record = {
        "update_index": int(update_index),
        "update_kind": str(update_kind),
        "viewpoint": int(viewpoint),
        "selected_view_indices": list(map(int, selected_view_indices)),
        "cnabu_observation_update_seconds": (
            None if cnabu_update_seconds is None else float(cnabu_update_seconds)
        ),
        "timing": graph_timing,
        "counts": counts,
        "display_tracking": display_state,
        "json_path": str(json_path) if json_path is not None else None,
        "png_path": str(png_path) if png_path is not None else None,
        "belief_png_path": str(belief_png_path) if belief_png_path is not None else None,
    }
    if ranked_advisory is not None:
        record["ranked_relation_advisory"] = {
            "available": bool(ranked_advisory.get("available", False)),
            "unavailable_reason": ranked_advisory.get("unavailable_reason"),
            "selected_target_node_id": ranked_advisory.get(
                "selected_target_node_id"
            ),
            "target_blockage_probability": ranked_advisory.get(
                "target_blockage_probability"
            ),
            "target_accessibility_probability": ranked_advisory.get(
                "target_accessibility_probability"
            ),
            "top_blocker_node_ids": [
                blocker["source_node_id"]
                for blocker in ranked_advisory.get("top_blockers", [])
            ],
            "timing_seconds": ranked_advisory.get("timing_seconds"),
            "executes_action": False,
        }
    if extra_record:
        record.update(dict(extra_record))
    summary["updates"].append(record)
    summary["updates_completed"] = int(len(summary["updates"]))
    print(json.dumps(record, sort_keys=True, default=json_default))
    return record, graph_window_open


def observe_belief_update(
    *,
    mem: ManipulationEnhancedMapping,
    previous_views: List[int],
    viewpoint: int,
    previous_map: Any,
    previous_semantic_map: Any,
) -> tuple[Any, Any, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    update_started = time.perf_counter()
    occupancy_distribution, semantic_concentration = mem.execute_observation(
        previous_views,
        int(viewpoint),
        previous_map,
        previous_semantic_map,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return occupancy_distribution, semantic_concentration, float(time.perf_counter() - update_started)


def observe_belief_update_with_optional_arm_motion(
    *,
    mem: ManipulationEnhancedMapping,
    previous_views: List[int],
    viewpoint: int,
    previous_map: Any,
    previous_semantic_map: Any,
    args: argparse.Namespace,
    summary: Dict[str, Any],
) -> tuple[Any, Any, float, Dict[str, Any]]:
    arm_motion = move_arm_for_observation(mem=mem, viewpoint=int(viewpoint), args=args)
    summary.setdefault("observation_arm_motion", []).append(arm_motion)
    occupancy_distribution, semantic_concentration, update_seconds = observe_belief_update(
        mem=mem,
        previous_views=previous_views,
        viewpoint=int(viewpoint),
        previous_map=previous_map,
        previous_semantic_map=previous_semantic_map,
    )
    return occupancy_distribution, semantic_concentration, update_seconds, arm_motion


def max_path_joint_delta(path: Sequence[Sequence[float]]) -> float:
    if not path:
        return 0.0
    configs = np.asarray([np.asarray(config[1:7], dtype=np.float64) for config in path], dtype=np.float64)
    if configs.ndim != 2 or configs.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(configs, axis=0))))


def plan_push_candidate(
    *,
    mem: ManipulationEnhancedMapping,
    previous_map: Any,
    previous_semantic_map: Any,
    num_points: int,
    ig_skip: int,
) -> tuple[Mapping[str, Any], int, int, float, float, float]:
    planning_started = time.perf_counter()
    push_candidates = mem.get_possible_maps_push(
        previous_map,
        previous_semantic_map,
        num_points=int(num_points),
    )
    planning_seconds = float(time.perf_counter() - planning_started)
    if push_candidates["paths"] is None:
        raise RuntimeError("MEM push planner did not return any feasible push paths")

    eval_started = time.perf_counter()
    best_viewpoint, best_push, best_push_ig = mem.eval_push_igs(
        push_candidates,
        previous_semantic_map,
        use_delta_H=True,
        skip=int(ig_skip),
    )
    eval_seconds = float(time.perf_counter() - eval_started)
    return (
        push_candidates,
        int(best_viewpoint),
        int(best_push),
        float(best_push_ig),
        planning_seconds,
        eval_seconds,
    )


def execute_planned_push(
    *,
    mem: ManipulationEnhancedMapping,
    push_candidates: Mapping[str, Any],
    best_viewpoint: int,
    best_push: int,
    best_push_ig: float,
    planning_seconds: float,
    eval_seconds: float,
) -> tuple[Any, Any, Dict[str, Any]]:
    object_positions_before, _ = mem.obj.update_obj_states(mem.current_obj_ids)
    selected_path = push_candidates["paths"][int(best_push)]
    selected_annotations = push_candidates["path_annotations"][int(best_push)]
    path_joint_delta = max_path_joint_delta(selected_path)

    execute_started = time.perf_counter()
    push_return_code, moved_positions = execute_push(
        mem,
        selected_path,
        path_annotations=selected_annotations,
    )
    execute_seconds = float(time.perf_counter() - execute_started)
    object_positions_after, _ = mem.obj.update_obj_states(mem.current_obj_ids)
    object_xy_displacement = float(
        sum(
            np.linalg.norm(np.asarray(before[:2]) - np.asarray(after[:2]))
            for before, after in zip(object_positions_before, object_positions_after)
        )
    )
    object_dropped = bool(mem.obj.check_all_object_drop(mem.current_obj_ids))

    predicted_map = push_candidates["possible_previous_maps"][int(best_push)][None]
    predicted_semantic_map = push_candidates["possible_semantic_maps"][int(best_push)][None]
    push_info = {
        "push_executed": True,
        "best_push_index": int(best_push),
        "best_post_push_viewpoint": int(best_viewpoint),
        "best_push_ig": float(best_push_ig),
        "push_return_code": int(push_return_code),
        "object_dropped": object_dropped,
        "num_candidate_paths": int(len(push_candidates["paths"])),
        "selected_path_waypoints": int(len(selected_path)),
        "selected_path_annotations": list(map(str, selected_annotations)),
        "selected_path_max_joint_delta": path_joint_delta,
        "moved_pose_samples": int(len(moved_positions)),
        "object_xy_displacement": object_xy_displacement,
        "planning_seconds": planning_seconds,
        "push_eval_seconds": eval_seconds,
        "push_execute_seconds": execute_seconds,
    }
    return predicted_map, predicted_semantic_map, push_info


def plan_and_execute_push(
    *,
    mem: ManipulationEnhancedMapping,
    previous_map: Any,
    previous_semantic_map: Any,
    num_points: int,
    ig_skip: int,
) -> tuple[Any, Any, int, Dict[str, Any]]:
    (
        push_candidates,
        best_viewpoint,
        best_push,
        best_push_ig,
        planning_seconds,
        eval_seconds,
    ) = plan_push_candidate(
        mem=mem,
        previous_map=previous_map,
        previous_semantic_map=previous_semantic_map,
        num_points=num_points,
        ig_skip=ig_skip,
    )
    predicted_map, predicted_semantic_map, push_info = execute_planned_push(
        mem=mem,
        push_candidates=push_candidates,
        best_viewpoint=best_viewpoint,
        best_push=best_push,
        best_push_ig=best_push_ig,
        planning_seconds=planning_seconds,
        eval_seconds=eval_seconds,
    )
    return predicted_map, predicted_semantic_map, int(best_viewpoint), push_info


def choose_observation_viewpoint(
    *,
    mem: ManipulationEnhancedMapping,
    previous_map: Any,
    previous_views: Sequence[int],
) -> tuple[int, float, float]:
    started = time.perf_counter()
    observation_igs, _ = get_igs_for_map(
        previous_map,
        mem.ig_calc,
        skip=1,
        use_alternative=True,
    )
    if len(previous_views) > 0:
        observation_igs[list(map(int, previous_views))] = 0
    viewpoint = int(observation_igs.argmax())
    return viewpoint, float(observation_igs.max()), float(time.perf_counter() - started)


def two_step_observation_ig(
    *,
    mem: ManipulationEnhancedMapping,
    previous_map: Any,
    previous_views: Sequence[int],
    first_viewpoint: int,
    first_ig: float,
) -> float:
    second_igs = get_subsequent_igs_for_map(previous_map, [int(first_viewpoint)], mem.ig_calc)
    if len(previous_views) > 0:
        second_igs[list(map(int, previous_views))] = 0
    return float(second_igs.max() + float(first_ig))


def mapped_fraction(mem: ManipulationEnhancedMapping, semantic_concentration: Any) -> float:
    sem_conf = mem.get_semantic_certainty(semantic_concentration)
    return float(mem.get_certainly_mapped_fraction(sem_conf, mem.prob_cutoff))


def run_policy_loop(
    *,
    args: argparse.Namespace,
    summary: Dict[str, Any],
    diagnostics_dir: Path,
    mode: str,
    splitter: Optional[LearnedCnabuComponentSplitter],
    display_tracker: SceneGraphDisplayTracker,
    mem: ManipulationEnhancedMapping,
    previous_map: Any,
    previous_semantic_map: Any,
    previous_views: List[int],
    graph_window_open: bool,
    window_name: str,
) -> bool:
    collision = False
    done_mapping = False
    fresh_push = False
    post_push_viewpoint: Optional[int] = None
    update_index = 0
    push_events: List[Dict[str, Any]] = []
    action_records: List[Dict[str, Any]] = []

    for step_index in range(int(args.action_budget)):
        step_started = time.perf_counter()
        policy_record: Dict[str, Any] = {
            "policy_step_index": int(step_index),
            "policy_loop": True,
            "policy_can_push": False,
            "policy_push_executed": False,
            "policy_collision": bool(collision),
        }

        if collision:
            policy_record["policy_action"] = "stopped_collision"
            action_records.append(policy_record)
            break

        if fresh_push:
            viewpoint = int(post_push_viewpoint if post_push_viewpoint is not None else 0)
            occupancy_distribution, semantic_concentration, update_seconds, arm_motion = (
                observe_belief_update_with_optional_arm_motion(
                    mem=mem,
                    previous_views=previous_views,
                    viewpoint=viewpoint,
                    previous_map=previous_map,
                    previous_semantic_map=previous_semantic_map,
                    args=args,
                    summary=summary,
                )
            )
            previous_map, previous_semantic_map = occupancy_distribution, semantic_concentration
            fresh_push = False
            certainty = mapped_fraction(mem, previous_semantic_map)
            done_mapping = bool(certainty >= mem.stopping_criterion)
            policy_record.update(
                {
                    "policy_action": "post_push_observe",
                    "viewpoint": int(viewpoint),
                    "mapped_fraction": certainty,
                    "step_seconds": float(time.perf_counter() - step_started),
                    "observation_arm_motion": arm_motion,
                }
            )
            _, graph_window_open = process_graph_update(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=mode,
                splitter=splitter,
                display_tracker=display_tracker,
                occupancy_distribution=previous_map,
                semantic_concentration=previous_semantic_map,
                update_index=update_index,
                update_kind="policy_post_push_observe",
                viewpoint=viewpoint,
                selected_view_indices=previous_views,
                cnabu_update_seconds=update_seconds,
                graph_window_open=graph_window_open,
                window_name=window_name,
                gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
                extra_record=policy_record,
            )
            update_index += 1
            action_records.append(policy_record)
            wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)
            if done_mapping and bool(args.stop_when_mapped):
                break
            continue

        viewpoint, max_obs_ig, vpp_seconds = choose_observation_viewpoint(
            mem=mem,
            previous_map=previous_map,
            previous_views=previous_views,
        )
        can_push = (
            not bool(args.disable_policy_push)
            and int(step_index) >= int(args.policy_push_start_step)
            and int(step_index) < int(args.action_budget) - 1
            and not done_mapping
        )
        policy_record.update(
            {
                "policy_can_push": bool(can_push),
                "best_observation_viewpoint": int(viewpoint),
                "best_observation_ig": float(max_obs_ig),
                "viewpoint_planning_seconds": float(vpp_seconds),
            }
        )

        should_push = False
        planned_push: Optional[tuple[Mapping[str, Any], int, int, float, float, float]] = None
        if can_push:
            best_observation_ig = two_step_observation_ig(
                mem=mem,
                previous_map=previous_map,
                previous_views=previous_views,
                first_viewpoint=viewpoint,
                first_ig=max_obs_ig,
            )
            policy_record["two_step_observation_ig"] = float(best_observation_ig)
            try:
                planned_push = plan_push_candidate(
                    mem=mem,
                    previous_map=previous_map,
                    previous_semantic_map=previous_semantic_map,
                    num_points=int(args.push_num_points),
                    ig_skip=int(args.push_ig_skip),
                )
                _, best_push_viewpoint, best_push, best_push_ig, planning_seconds, eval_seconds = planned_push
                should_push = bool(float(best_push_ig) > float(best_observation_ig))
                if (
                    args.force_policy_push_step is not None
                    and int(step_index) == int(args.force_policy_push_step)
                ):
                    should_push = True
                policy_record.update(
                    {
                        "best_push_index": int(best_push),
                        "best_push_viewpoint": int(best_push_viewpoint),
                        "best_push_ig": float(best_push_ig),
                        "push_planning_seconds": float(planning_seconds),
                        "push_eval_seconds": float(eval_seconds),
                        "push_beats_observation": bool(float(best_push_ig) > float(best_observation_ig)),
                        "push_forced_by_cli": bool(
                            args.force_policy_push_step is not None
                            and int(step_index) == int(args.force_policy_push_step)
                        ),
                    }
                )
            except RuntimeError as exc:
                policy_record["push_planning_error"] = str(exc)
                should_push = False

        if should_push and planned_push is not None:
            push_candidates, best_push_viewpoint, best_push, best_push_ig, planning_seconds, eval_seconds = planned_push
            previous_map, previous_semantic_map, push_info = execute_planned_push(
                mem=mem,
                push_candidates=push_candidates,
                best_viewpoint=best_push_viewpoint,
                best_push=best_push,
                best_push_ig=best_push_ig,
                planning_seconds=planning_seconds,
                eval_seconds=eval_seconds,
            )
            collision = bool(push_info.get("object_dropped", False))
            fresh_push = not collision
            post_push_viewpoint = int(best_push_viewpoint)
            previous_views.clear()
            summary["push_executed"] = True
            summary["push_info"] = push_info
            push_event = {
                **push_info,
                "policy_step_index": int(step_index),
                "update_index": int(update_index),
            }
            push_events.append(push_event)
            certainty = mapped_fraction(mem, previous_semantic_map)
            done_mapping = bool(certainty >= mem.stopping_criterion)
            policy_record.update(
                {
                    "policy_action": "push_predicted",
                    "policy_push_executed": True,
                    "viewpoint": int(best_push_viewpoint),
                    "mapped_fraction": certainty,
                    "step_seconds": float(time.perf_counter() - step_started),
                }
            )
            _, graph_window_open = process_graph_update(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=mode,
                splitter=splitter,
                display_tracker=display_tracker,
                occupancy_distribution=previous_map,
                semantic_concentration=previous_semantic_map,
                update_index=update_index,
                update_kind="policy_push_predicted",
                viewpoint=int(best_push_viewpoint),
                selected_view_indices=previous_views,
                cnabu_update_seconds=None,
                graph_window_open=graph_window_open,
                window_name=window_name,
                gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
                extra_record=policy_record,
            )
            update_index += 1
            action_records.append(policy_record)
            wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)
            if done_mapping and bool(args.stop_when_mapped):
                break
            continue

        occupancy_distribution, semantic_concentration, update_seconds, arm_motion = (
            observe_belief_update_with_optional_arm_motion(
                mem=mem,
                previous_views=previous_views,
                viewpoint=viewpoint,
                previous_map=previous_map,
                previous_semantic_map=previous_semantic_map,
                args=args,
                summary=summary,
            )
        )
        previous_map, previous_semantic_map = occupancy_distribution, semantic_concentration
        certainty = mapped_fraction(mem, previous_semantic_map)
        done_mapping = bool(certainty >= mem.stopping_criterion)
        policy_record.update(
            {
                "policy_action": "observe",
                "viewpoint": int(viewpoint),
                "mapped_fraction": certainty,
                "step_seconds": float(time.perf_counter() - step_started),
                "observation_arm_motion": arm_motion,
            }
        )
        _, graph_window_open = process_graph_update(
            args=args,
            summary=summary,
            diagnostics_dir=diagnostics_dir,
            mode=mode,
            splitter=splitter,
            display_tracker=display_tracker,
            occupancy_distribution=previous_map,
            semantic_concentration=previous_semantic_map,
            update_index=update_index,
            update_kind="policy_observe",
            viewpoint=int(viewpoint),
            selected_view_indices=previous_views,
            cnabu_update_seconds=update_seconds,
            graph_window_open=graph_window_open,
            window_name=window_name,
            gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
            extra_record=policy_record,
        )
        update_index += 1
        action_records.append(policy_record)
        wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)
        if done_mapping and bool(args.stop_when_mapped):
            break

    summary["policy_actions"] = action_records
    summary["push_events"] = push_events
    summary["num_pushes_executed"] = int(len(push_events))
    summary["policy_steps_completed"] = int(len(action_records))
    return graph_window_open


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# MEM CNABU Scene Graph Live Demo",
        "",
        f"Created: `{summary['created_at']}`",
        f"Host: `{summary['host']}`",
        f"Command: `{summary['command']}`",
        f"Mode: `{summary['mode']}`",
        f"Checkpoint: `{summary.get('checkpoint')}`",
        f"Ranked relation advisory: `{summary.get('ranked_relation_advisory_enabled', False)}`",
        f"Ranked relation checkpoint: `{summary.get('ranked_relation_checkpoint')}`",
        f"Ranked relation config: `{summary.get('ranked_relation_config')}`",
        f"Ranked relation threshold: `{summary.get('ranked_relation_threshold')}`",
        f"Updates requested: `{summary['updates_requested']}`",
        f"Updates completed: `{summary['updates_completed']}`",
        f"Render PyBullet: `{summary['render']}`",
        f"Show graph window: `{summary['show_graph']}`",
        f"Show belief-only panel: `{summary.get('show_belief_panel', False)}`",
        f"Show GT panel: `{summary['show_gt_panel']}`",
        f"Policy loop: `{summary['policy_loop']}`",
        f"Action budget: `{summary['action_budget']}`",
        f"Move arm for observations: `{summary['move_arm_for_observe']}`",
        f"Push enabled: `{summary['push_enabled']}`",
        f"Push executed: `{summary.get('push_executed', False)}`",
        f"Pushes executed: `{summary.get('num_pushes_executed', int(bool(summary.get('push_executed'))))}`",
        f"Diagnostics enabled: `{summary['save_diagnostics']}`",
        f"Diagnostics dir: `{summary.get('diagnostics_dir')}`",
        "",
        "## Safety",
        "",
        f"- GT loaded for graph input: `{summary['gt_loaded']}`",
        f"- GT diagnostic panel loaded: `{summary.get('gt_diagnostic_loaded', False)}`",
        f"- Simulator instance labels used for graph input: `{summary['simulator_instance_labels_used_for_graph_input']}`",
        f"- D3G runtime helpers imported: `{summary['d3g_runtime_helpers_imported']}`",
        f"- Training run: `{summary['training_run']}`",
        f"- Dataset export written: `{summary['dataset_export_written']}`",
        f"- Checkpoint/model artifact written: `{summary['checkpoint_model_artifact_written']}`",
        f"- Ranked advisory executes action: `{summary.get('ranked_relation_executes_action', False)}`",
        "",
        "## Live Tensor Shapes",
        "",
    ]
    for key, value in summary.get("live_tensor_shapes", {}).items():
        lines.append(f"- {key}: `{value}`")
    if summary.get("move_arm_for_observe"):
        motion_records = list(summary.get("observation_arm_motion", []))
        success_count = sum(1 for record in motion_records if bool(record.get("arm_observe_motion_success", False)))
        lines.extend(["", "## Observation Arm Motion", ""])
        lines.append(f"- Target link: `{summary.get('observe_arm_target_link')}`")
        lines.append(f"- Successful moves: `{success_count}/{len(motion_records)}`")
        lines.append(f"- Return home after observation: `{not bool(summary.get('observe_arm_keep_pose'))}`")
        if motion_records:
            lines.append("")
            lines.append("| Observation | Success | View target | Move samples | Return home | Note |")
            lines.append("| ---: | --- | --- | ---: | --- | --- |")
            for index, record in enumerate(motion_records):
                target_position = record.get("arm_observe_motion_target_position")
                target_text = "None" if target_position is None else [round(float(v), 3) for v in target_position]
                lines.append(
                    f"| {index} | `{record.get('arm_observe_motion_success')}` | `{target_text}` | "
                    f"{record.get('arm_observe_motion_samples', 0)} | "
                    f"`{record.get('home_return_success', None)}` | "
                    f"`{record.get('arm_observe_motion_error', '')}` |"
                )
    if summary.get("push_info"):
        lines.extend(["", "## Push", ""])
        push_info = summary["push_info"]
        for key in (
            "push_executed",
            "best_push_index",
            "best_post_push_viewpoint",
            "best_push_ig",
            "push_return_code",
            "object_dropped",
            "num_candidate_paths",
            "selected_path_waypoints",
            "selected_path_max_joint_delta",
            "moved_pose_samples",
            "object_xy_displacement",
            "planning_seconds",
            "push_eval_seconds",
            "push_execute_seconds",
        ):
            lines.append(f"- {key}: `{push_info.get(key)}`")
    if summary.get("push_events"):
        lines.extend(["", "## Push Events", ""])
        lines.append("| Event | Step | Update | Best view | IG | Samples | Object XY |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for event_index, event in enumerate(summary.get("push_events", []), start=1):
            lines.append(
                f"| {event_index} | {event.get('policy_step_index')} | {event.get('update_index')} | "
                f"{event.get('best_post_push_viewpoint')} | {float(event.get('best_push_ig', 0.0)):.3f} | "
                f"{event.get('moved_pose_samples')} | {float(event.get('object_xy_displacement', 0.0)):.4f} |"
            )
    if summary.get("policy_actions"):
        lines.extend(["", "## Policy Actions", ""])
        lines.append("| Step | Action | Viewpoint | Mapped | Obs IG | Push IG | Push? |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | --- |")
        for action in summary.get("policy_actions", []):
            lines.append(
                f"| {action.get('policy_step_index')} | `{action.get('policy_action')}` | "
                f"{action.get('viewpoint', action.get('best_observation_viewpoint'))} | "
                f"{float(action.get('mapped_fraction', 0.0)):.3f} | "
                f"{float(action.get('best_observation_ig', 0.0)):.3f} | "
                f"{float(action.get('best_push_ig', 0.0)):.3f} | "
                f"`{bool(action.get('policy_push_executed', False))}` |"
            )
    lines.extend(["", "## Updates", ""])
    lines.append("| Update | Kind | Viewpoint | Nodes | Edges | JSON-safe | Graph sec | JSON | PNG | Belief PNG |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |")
    for record in summary.get("updates", []):
        counts = record["counts"]
        lines.append(
            f"| {record['update_index']} | `{record.get('update_kind')}` | {record['viewpoint']} | "
            f"{counts['nodes']} | {counts['edges']} | "
            f"`{counts['json_safe']}` | {record['timing'].get('total_graph_generation', 0.0):.4f} | "
            f"`{record.get('json_path')}` | `{record.get('png_path')}` | "
            f"`{record.get('belief_png_path')}` |"
        )
    lines.extend(["", "## Recommendation", "", str(summary["recommendation"]), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-graph-mode",
        choices=("split_off", "split_on_2d_candidate", "learned_component_splitter"),
        default="learned_component_splitter",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--viewpoints", default=None, help="Comma-separated viewpoint ids. Defaults to uniform ids.")
    parser.add_argument("--render", action="store_true", help="Open the PyBullet GUI.")
    parser.add_argument("--show-graph", action="store_true", help="Open/update the live OpenCV scene-graph visualization.")
    parser.add_argument(
        "--full-workspace-view",
        action="store_true",
        help="Show the complete 140x200 MEM workspace instead of the default fixed shelf-focused viewport.",
    )
    parser.add_argument(
        "--show-belief-panel",
        action="store_true",
        help="Append a CNABU belief-map-only panel beside the graph view; no graph nodes or edges are drawn on it.",
    )
    parser.add_argument(
        "--show-gt-panel",
        action="store_true",
        help="Show the coordinate-aligned GT instance scene graph beside the prediction (evaluation only).",
    )
    parser.add_argument("--enable-push", action="store_true", help="Run observe, push-predicted, post-push observe.")
    parser.add_argument("--policy-loop", action="store_true", help="Run a bounded MEM observe-vs-push policy loop.")
    parser.add_argument("--action-budget", type=int, default=6, help="Policy-loop action budget. MEM's full default is 40.")
    parser.add_argument("--policy-push-start-step", type=int, default=3)
    parser.add_argument("--disable-policy-push", action="store_true")
    parser.add_argument("--force-policy-push-step", type=int, default=None)
    parser.add_argument("--stop-when-mapped", action="store_true")
    parser.add_argument(
        "--move-arm-for-observe",
        action="store_true",
        help="Before each observation, move the UR5 dummy_camera_link to the selected camera-array pose.",
    )
    parser.add_argument("--observe-arm-target-link", default="dummy_camera_link")
    parser.add_argument("--observe-arm-keep-pose", action="store_true")
    parser.add_argument("--observe-arm-full-plan", action="store_true")
    parser.add_argument("--observe-arm-pause-sec", type=float, default=0.0)
    parser.add_argument("--observe-arm-verbose", action="store_true")
    parser.add_argument("--save-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-dir", type=Path, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.1)
    parser.add_argument("--hold-sec", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", help="Device for learned splitter inference.")
    parser.add_argument(
        "--enable-ranked-relations",
        action="store_true",
        help=(
            "Opt in to advisory target-local blocker ranking; this never executes "
            "a removal or manipulation action."
        ),
    )
    parser.add_argument(
        "--ranked-relation-config",
        type=Path,
        default=DEFAULT_RANKED_RELATION_CONFIG,
    )
    parser.add_argument(
        "--ranked-relation-checkpoint",
        type=Path,
        default=DEFAULT_RANKED_RELATION_CHECKPOINT,
    )
    parser.add_argument(
        "--ranked-relation-threshold",
        type=float,
        default=DEFAULT_RANKED_RELATION_THRESHOLD,
    )
    parser.add_argument("--ranked-relation-device", default="cpu")
    parser.add_argument("--ranked-target-node-id", type=int, default=None)
    parser.add_argument("--ranked-top-k", type=int, default=3)
    parser.add_argument("--max-obj-num", type=int, default=12)
    parser.add_argument("--occupancy-threshold", type=float, default=0.35)
    parser.add_argument("--push-num-points", type=int, default=30)
    parser.add_argument("--push-ig-skip", type=int, default=5)
    parser.add_argument("--post-push-viewpoint", type=int, default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if bool(args.policy_loop) and bool(args.enable_push):
        raise ValueError("--policy-loop and --enable-push are separate modes; choose one")
    if bool(args.enable_ranked_relations) and args.scene_graph_mode != "learned_component_splitter":
        raise ValueError(
            "--enable-ranked-relations requires learned_component_splitter runtime nodes"
        )
    if not 0.0 <= float(args.ranked_relation_threshold) <= 1.0:
        raise ValueError("--ranked-relation-threshold must be in [0,1]")
    if int(args.ranked_top_k) <= 0:
        raise ValueError("--ranked-top-k must be positive")
    updates = int(args.action_budget) if bool(args.policy_loop) else (3 if bool(args.enable_push) else int(args.updates))
    viewpoints = parse_viewpoints(args.viewpoints, updates=updates)
    diagnostics_dir = args.diagnostics_dir or default_diagnostics_dir()
    if bool(args.save_diagnostics):
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

    splitter: Optional[LearnedCnabuComponentSplitter] = None
    d3g_runtime_helpers_imported = False
    checkpoint_load_seconds = 0.0
    if args.scene_graph_mode == "learned_component_splitter":
        splitter = LearnedCnabuComponentSplitter(args.checkpoint, device=args.device)
        checkpoint_load_seconds = float(splitter.load_seconds)
        d3g_runtime_helpers_imported = True

    ranked_relation_runtime: Optional[RankedRelationAdvisoryRuntime] = None
    ranked_relation_checkpoint_load_seconds = 0.0
    ranked_relation_checkpoint_load: Optional[Mapping[str, Any]] = None
    ranked_relation_bridge_resource_after_load: Optional[Mapping[str, Any]] = None
    if bool(args.enable_ranked_relations):
        ranked_relation_runtime = RankedRelationAdvisoryRuntime(
            config_path=args.ranked_relation_config.resolve(),
            checkpoint_path=args.ranked_relation_checkpoint.resolve(),
            threshold=float(args.ranked_relation_threshold),
            top_k=int(args.ranked_top_k),
            target_node_id=args.ranked_target_node_id,
            device=str(args.ranked_relation_device),
        )
        ranked_relation_runtime.start()
        ranked_relation_checkpoint_load_seconds = ranked_relation_runtime.load_seconds
        ranked_relation_checkpoint_load = ranked_relation_runtime.checkpoint_load
        ranked_relation_bridge_resource_after_load = (
            ranked_relation_runtime.bridge_resource_after_load
        )
    setattr(args, "_ranked_relation_runtime", ranked_relation_runtime)

    summary: Dict[str, Any] = {
        "schema": "mem_cnabu_scene_graph_live_demo_v0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        "mode": str(args.scene_graph_mode),
        "checkpoint": str(args.checkpoint) if args.scene_graph_mode == "learned_component_splitter" else None,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "ranked_relation_advisory_enabled": bool(args.enable_ranked_relations),
        "ranked_relation_executes_action": False,
        "ranked_relation_config": (
            str(args.ranked_relation_config) if args.enable_ranked_relations else None
        ),
        "ranked_relation_checkpoint": (
            str(args.ranked_relation_checkpoint)
            if args.enable_ranked_relations
            else None
        ),
        "ranked_relation_checkpoint_load": (
            dict(ranked_relation_checkpoint_load)
            if ranked_relation_checkpoint_load is not None
            else None
        ),
        "ranked_relation_checkpoint_load_seconds": (
            ranked_relation_checkpoint_load_seconds
        ),
        "ranked_relation_bridge_resource_after_load": (
            dict(ranked_relation_bridge_resource_after_load)
            if ranked_relation_bridge_resource_after_load is not None
            else None
        ),
        "ranked_relation_threshold": float(args.ranked_relation_threshold),
        "ranked_target_node_id": args.ranked_target_node_id,
        "ranked_top_k": int(args.ranked_top_k),
        "updates_requested": updates,
        "updates_completed": 0,
        "viewpoints": viewpoints,
        "render": bool(args.render),
        "show_graph": bool(args.show_graph),
        "full_workspace_view": bool(args.full_workspace_view),
        "graph_view_xyxy": (
            None if bool(args.full_workspace_view) else list(DEFAULT_SHELF_VIEW_XYXY)
        ),
        "show_belief_panel": bool(args.show_belief_panel),
        "show_gt_panel": bool(args.show_gt_panel),
        "save_diagnostics": bool(args.save_diagnostics),
        "diagnostics_dir": str(diagnostics_dir) if bool(args.save_diagnostics) else None,
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "gt_loaded": False,
        "gt_diagnostic_loaded": False,
        "simulator_instance_labels_used_for_graph_input": False,
        "d3g_runtime_helpers_imported": bool(d3g_runtime_helpers_imported),
        "training_run": False,
        "dataset_export_written": False,
        "checkpoint_model_artifact_written": False,
        "push_enabled": bool(args.enable_push),
        "policy_loop": bool(args.policy_loop),
        "action_budget": int(args.action_budget),
        "policy_push_start_step": int(args.policy_push_start_step),
        "disable_policy_push": bool(args.disable_policy_push),
        "force_policy_push_step": args.force_policy_push_step,
        "stop_when_mapped": bool(args.stop_when_mapped),
        "move_arm_for_observe": bool(args.move_arm_for_observe),
        "observe_arm_target_link": str(args.observe_arm_target_link),
        "observe_arm_keep_pose": bool(args.observe_arm_keep_pose),
        "observe_arm_full_plan": bool(args.observe_arm_full_plan),
        "observe_arm_pause_sec": float(args.observe_arm_pause_sec),
        "push_executed": False,
        "push_info": None,
        "push_events": [],
        "num_pushes_executed": 0,
        "policy_actions": [],
        "policy_steps_completed": 0,
        "observation_arm_motion": [],
        "push_num_points": int(args.push_num_points),
        "push_ig_skip": int(args.push_ig_skip),
        "post_push_viewpoint_override": args.post_push_viewpoint,
        "updates": [],
        "live_tensor_shapes": {},
    }

    mem: Optional[ManipulationEnhancedMapping] = None
    graph_window_open = False
    window_name = "MEM live spatial scene graph"
    display_tracker = SceneGraphDisplayTracker()
    try:
        init_started = time.perf_counter()
        mem = ManipulationEnhancedMapping(
            render=bool(args.render),
            show_vis=False,
            use_uncertainty_informed_sampling=False,
            use_ycb=True,
            max_obj_num=int(args.max_obj_num),
            max_occupancy_threshold=float(args.occupancy_threshold),
        )
        if ranked_relation_runtime is not None:
            ranked_relation_runtime.environment = mem
        summary["environment_init_seconds"] = float(time.perf_counter() - init_started)
        reset_started = time.perf_counter()
        mem.reset_env(occupancy_threshold=float(args.occupancy_threshold))
        summary["environment_reset_seconds"] = float(time.perf_counter() - reset_started)

        map_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        previous_map, previous_semantic_map = mem.map_completion_model.dp.get_initial_map(
            torch.ones((1, 1, 204, 120, 200), device=map_device)
        )
        previous_views: List[int] = []

        if bool(args.policy_loop):
            graph_window_open = run_policy_loop(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=str(args.scene_graph_mode),
                splitter=splitter,
                display_tracker=display_tracker,
                mem=mem,
                previous_map=previous_map,
                previous_semantic_map=previous_semantic_map,
                previous_views=previous_views,
                graph_window_open=graph_window_open,
                window_name=window_name,
            )
        elif bool(args.enable_push):
            initial_viewpoint = int(viewpoints[0])
            occupancy_distribution, semantic_concentration, update_seconds, arm_motion = (
                observe_belief_update_with_optional_arm_motion(
                    mem=mem,
                    previous_views=previous_views,
                    viewpoint=initial_viewpoint,
                    previous_map=previous_map,
                    previous_semantic_map=previous_semantic_map,
                    args=args,
                    summary=summary,
                )
            )
            previous_map, previous_semantic_map = occupancy_distribution, semantic_concentration
            _, graph_window_open = process_graph_update(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=str(args.scene_graph_mode),
                splitter=splitter,
                display_tracker=display_tracker,
                occupancy_distribution=occupancy_distribution,
                semantic_concentration=semantic_concentration,
                update_index=0,
                update_kind="observe_initial",
                viewpoint=initial_viewpoint,
                selected_view_indices=previous_views,
                cnabu_update_seconds=update_seconds,
                graph_window_open=graph_window_open,
                window_name=window_name,
                gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
                extra_record={"observation_arm_motion": arm_motion},
            )
            wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)

            predicted_map, predicted_semantic_map, best_viewpoint, push_info = plan_and_execute_push(
                mem=mem,
                previous_map=previous_map,
                previous_semantic_map=previous_semantic_map,
                num_points=int(args.push_num_points),
                ig_skip=int(args.push_ig_skip),
            )
            summary["push_executed"] = True
            summary["push_info"] = push_info
            summary["push_events"] = [{**push_info, "policy_step_index": None, "update_index": 1}]
            summary["num_pushes_executed"] = 1

            previous_views.clear()
            previous_map, previous_semantic_map = predicted_map, predicted_semantic_map
            _, graph_window_open = process_graph_update(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=str(args.scene_graph_mode),
                splitter=splitter,
                display_tracker=display_tracker,
                occupancy_distribution=previous_map,
                semantic_concentration=previous_semantic_map,
                update_index=1,
                update_kind="push_predicted",
                viewpoint=best_viewpoint,
                selected_view_indices=previous_views,
                cnabu_update_seconds=None,
                graph_window_open=graph_window_open,
                window_name=window_name,
                gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
            )
            wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)

            post_push_viewpoint = (
                int(args.post_push_viewpoint)
                if args.post_push_viewpoint is not None
                else int(best_viewpoint)
            )
            occupancy_distribution, semantic_concentration, update_seconds, arm_motion = (
                observe_belief_update_with_optional_arm_motion(
                    mem=mem,
                    previous_views=previous_views,
                    viewpoint=post_push_viewpoint,
                    previous_map=previous_map,
                    previous_semantic_map=previous_semantic_map,
                    args=args,
                    summary=summary,
                )
            )
            previous_map, previous_semantic_map = occupancy_distribution, semantic_concentration
            _, graph_window_open = process_graph_update(
                args=args,
                summary=summary,
                diagnostics_dir=diagnostics_dir,
                mode=str(args.scene_graph_mode),
                splitter=splitter,
                display_tracker=display_tracker,
                occupancy_distribution=occupancy_distribution,
                semantic_concentration=semantic_concentration,
                update_index=2,
                update_kind="post_push_observe",
                viewpoint=post_push_viewpoint,
                selected_view_indices=previous_views,
                cnabu_update_seconds=update_seconds,
                graph_window_open=graph_window_open,
                window_name=window_name,
                gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
                extra_record={"observation_arm_motion": arm_motion},
            )
            wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)
        else:
            for update_index, viewpoint in enumerate(viewpoints):
                occupancy_distribution, semantic_concentration, update_seconds, arm_motion = (
                    observe_belief_update_with_optional_arm_motion(
                        mem=mem,
                        previous_views=previous_views,
                        viewpoint=int(viewpoint),
                        previous_map=previous_map,
                        previous_semantic_map=previous_semantic_map,
                        args=args,
                        summary=summary,
                    )
                )
                previous_map, previous_semantic_map = occupancy_distribution, semantic_concentration
                _, graph_window_open = process_graph_update(
                    args=args,
                    summary=summary,
                    diagnostics_dir=diagnostics_dir,
                    mode=str(args.scene_graph_mode),
                    splitter=splitter,
                    display_tracker=display_tracker,
                    occupancy_distribution=occupancy_distribution,
                    semantic_concentration=semantic_concentration,
                    update_index=update_index,
                    update_kind="observe",
                    viewpoint=int(viewpoint),
                    selected_view_indices=previous_views,
                    cnabu_update_seconds=update_seconds,
                    graph_window_open=graph_window_open,
                    window_name=window_name,
                    gt_data=get_gt_diagnostic_data(mem=mem, args=args, summary=summary),
                    extra_record={"observation_arm_motion": arm_motion},
                )
                wait_after_update(sleep_sec=float(args.sleep_sec), graph_window_open=graph_window_open)

        all_counts = [record["counts"] for record in summary["updates"]]
        nonempty_graphs = all(
            counts["nodes"] > 0 and counts["edges"] > 0 and counts["json_safe"] for counts in all_counts
        )
        expected_push_kinds = ["observe_initial", "push_predicted", "post_push_observe"]
        push_kinds = [str(record.get("update_kind")) for record in summary["updates"]]
        if bool(args.policy_loop) and not (
            summary["updates_completed"] > 1
            and nonempty_graphs
            and (
                bool(args.disable_policy_push)
                or summary.get("num_pushes_executed", 0) > 0
                or int(args.action_budget) <= int(args.policy_push_start_step)
            )
        ):
            summary["recommendation"] = "needs longer/stronger policy-loop run"
        elif bool(args.enable_push) and not (
            summary["push_executed"] and push_kinds == expected_push_kinds and nonempty_graphs
        ):
            summary["recommendation"] = "needs fixes"
        elif nonempty_graphs:
            summary["recommendation"] = "ready for live demo"
        elif any(counts["nodes"] > 0 for counts in all_counts):
            summary["recommendation"] = "needs fixes"
        else:
            summary["recommendation"] = "needs stronger checkpoint"

        if bool(args.save_diagnostics):
            summary_json = diagnostics_dir / "run_summary.json"
            summary_md = diagnostics_dir / "summary.md"
            write_json(summary_json, summary)
            write_summary(summary_md, summary)
            print(json.dumps({"summary": str(summary_md), "summary_json": str(summary_json)}, sort_keys=True))

        if float(args.hold_sec) > 0.0:
            deadline = time.time() + float(args.hold_sec)
            while time.time() < deadline:
                if graph_window_open:
                    key = cv2.waitKey(100)
                    if key in (27, ord("q")):
                        break
                else:
                    time.sleep(min(0.25, max(0.0, deadline - time.time())))
        return 0
    finally:
        if graph_window_open:
            cv2.destroyWindow(window_name)
        if ranked_relation_runtime is not None:
            ranked_relation_runtime.close()
        if mem is not None:
            mem.close()


if __name__ == "__main__":
    raise SystemExit(main())
