"""Benchmark cached no-GT CNABU attribution over executable MEM cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch

from scene_graph_mem.relations.belief_occlusion import (
    build_unresolved_uncertainty_field,
    hidden_uncertainty_components,
)
from scene_graph_mem.relations.occlusion_attribution import (
    TorchHiddenUncertaintyCache,
)
from scene_graph_mem.relations.path_aligned_features import (
    reconstruct_sparse_node_voxel_support,
)
from shelf_gym.scripts.audit_cnabu_occlusion_attribution import (
    DEFAULT_CAMERA_PATH,
    DEFAULT_CNABU_ROOT,
    DEFAULT_NODE_ROOT,
    InfoGainEval,
    _parse_indices,
)
from shelf_gym.utils.cnabu_occlusion_attribution import (
    dense_supports_from_sparse_indices,
    info_gain_raycast_to_canonical_zyx,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
THESIS_ROOT = REPO_ROOT.parent
SCENE_GRAPH_ROOT = THESIS_ROOT / "scene_graph_mem"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _timed_score(
    cache: TorchHiddenUncertaintyCache,
    rays: np.ndarray,
    *,
    revision: str,
    device: str,
) -> Tuple[Dict[str, np.ndarray], float]:
    _synchronize(device)
    started = time.perf_counter()
    values = cache.score(rays, belief_revision=revision)
    _synchronize(device)
    return values, float(time.perf_counter() - started)


def _load_runtime_inputs(
    *,
    sample_id: str,
    cnabu_root: Path,
    node_root: Path,
    occupancy_threshold: float,
) -> Dict[str, Any]:
    cnabu_path = cnabu_root / "samples" / sample_id / "pre_action" / "cnabu_hms.npz"
    node_path = node_root / "samples" / sample_id / "pre_action" / "node_masks.npz"
    for path in (cnabu_path, node_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(cnabu_path, allow_pickle=False) as data:
        occupancy = np.asarray(data["occupancy_mean"], dtype=np.float32)
        epistemic = np.asarray(data["occupancy_epistemic"], dtype=np.float32)
        semantic = np.asarray(data["semantic_mean"], dtype=np.float32)
        vacuity = np.asarray(data["semantic_vacuity"], dtype=np.float32)
        crop_rows = tuple(int(value) for value in data["crop_rows"].tolist())
    with np.load(node_path, allow_pickle=False) as data:
        node_masks = np.asarray(data["node_masks"], dtype=bool)
        node_classes = np.asarray(data["node_semantic_labels"], dtype=np.int64)
        node_crop_rows = tuple(int(value) for value in data["crop_rows"].tolist())
        node_source = str(np.asarray(data["node_source"]).item())
    if crop_rows != node_crop_rows:
        raise ValueError("CNABU and learned-node crops differ")
    if node_source != "learned_component_splitter":
        raise ValueError("runtime benchmark requires learned splitter nodes")
    sparse = reconstruct_sparse_node_voxel_support(
        occupancy,
        semantic,
        node_masks,
        node_classes,
        crop_rows=crop_rows,
        occupancy_threshold=float(occupancy_threshold),
    )
    supports = dense_supports_from_sparse_indices(
        sparse.indices_zyx, grid_shape_zyx=sparse.grid_shape_zyx
    )
    field = build_unresolved_uncertainty_field(
        occupancy, epistemic, vacuity
    )
    components = {
        "occupancy_epistemic": field.occupancy_epistemic * field.lambda_occ,
        "semantic_vacuity": field.semantic_vacuity * field.lambda_sem,
        "total": field.total,
    }
    return {
        "occupancy": occupancy,
        "components": components,
        "supports": supports,
        "crop_rows": crop_rows,
        "sparse": sparse,
        "cnabu_path": cnabu_path,
        "node_path": node_path,
    }


def _camera_scaling(cumulative_seconds: Sequence[float], counts: Sequence[int]) -> Dict[str, Any]:
    result = {}
    for count in counts:
        if int(count) <= 0 or int(count) > len(cumulative_seconds):
            raise ValueError("camera scaling counts must lie inside measured workload")
        result[str(int(count))] = {
            "wall_seconds": float(cumulative_seconds[int(count) - 1]),
            "seconds_per_camera": float(
                cumulative_seconds[int(count) - 1] / int(count)
            ),
        }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sample-id", type=str, default="0/000000098")
    parser.add_argument("--camera-indices", type=str, default="0:300")
    parser.add_argument("--ray-subsample", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--source-batch-size", type=int, default=4)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--node-scaling-repeats", type=int, default=3)
    parser.add_argument("--cnabu-root", type=Path, default=DEFAULT_CNABU_ROOT)
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT)
    parser.add_argument("--camera-path", type=Path, default=DEFAULT_CAMERA_PATH)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=THESIS_ROOT
        / "thesis_records/diagnostics/"
        "2026-07-16_cnabu_belief_occlusion_stage0_baseline_smoke_v5/"
        "baseline_summary.json",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_path))
    device = str(torch.device(args.device))
    if torch.device(device).type != "cuda":
        raise ValueError("deployability benchmark requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    # PyTorch 2.9's memory-stat API rejects a lazily uninitialized CUDA
    # ``torch.device`` even though ``is_available`` is true.  Establish the
    # selected context explicitly before resetting its counters.
    torch.cuda.set_device(torch.device(device))
    camera_path = args.camera_path.resolve()
    baseline_path = args.baseline_summary.resolve()
    for path in (camera_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    load_started = time.perf_counter()
    inputs = _load_runtime_inputs(
        sample_id=str(args.sample_id),
        cnabu_root=args.cnabu_root.resolve(),
        node_root=args.node_root.resolve(),
        occupancy_threshold=float(args.occupancy_threshold),
    )
    input_load_seconds = float(time.perf_counter() - load_started)
    revision = "{}:offline-pre-action-belief".format(args.sample_id)
    cache_started = time.perf_counter()
    cache = TorchHiddenUncertaintyCache(
        occupancy_mean=inputs["occupancy"],
        component_masses=inputs["components"],
        source_supports=inputs["supports"],
        belief_revision=revision,
        support_provenance="runtime_cnabu_learned_component_splitter",
        device=device,
        source_batch_size=int(args.source_batch_size),
    )
    _synchronize(device)
    cache_construction_seconds = float(time.perf_counter() - cache_started)
    info_gain = InfoGainEval(
        str(camera_path),
        subsample=int(args.ray_subsample),
        occupancy_thold=0.95,
        cached=False,
    )
    camera_indices = _parse_indices(
        args.camera_indices, upper_bound=len(info_gain.camera_matrices)
    )

    # One warm-up exercises allocations before measured camera scaling.
    warm_raw = info_gain.get_raycast(camera_idx=int(camera_indices[0]))
    warm_rays = info_gain_raycast_to_canonical_zyx(
        warm_raw,
        grid_shape_zyx=inputs["occupancy"].shape,
        crop_rows=inputs["crop_rows"],
        raw_shape_hw=(140, 200),
    )
    _timed_score(cache, warm_rays, revision=revision, device=device)
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    per_camera = []
    score_values = {name: [] for name in inputs["components"]}
    cumulative_score_seconds = []
    accumulated = 0.0
    for camera_index in camera_indices:
        ray_started = time.perf_counter()
        raw = info_gain.get_raycast(camera_idx=int(camera_index))
        ray_generation_seconds = float(time.perf_counter() - ray_started)
        conversion_started = time.perf_counter()
        rays = info_gain_raycast_to_canonical_zyx(
            raw,
            grid_shape_zyx=inputs["occupancy"].shape,
            crop_rows=inputs["crop_rows"],
            raw_shape_hw=(140, 200),
        )
        conversion_seconds = float(time.perf_counter() - conversion_started)
        values, score_seconds = _timed_score(
            cache, rays, revision=revision, device=device
        )
        accumulated += score_seconds
        cumulative_score_seconds.append(accumulated)
        for name in score_values:
            score_values[name].append(values[name])
        per_camera.append(
            {
                "camera_index": int(camera_index),
                "ray_generation_seconds": ray_generation_seconds,
                "conversion_seconds": conversion_seconds,
                "score_seconds": score_seconds,
                "valid_ray_sample_count": int(np.all(rays >= 0, axis=-1).sum()),
            }
        )

    # Correctness and CPU references use one measured executable camera.
    cpu_started = time.perf_counter()
    cpu = hidden_uncertainty_components(
        warm_rays,
        inputs["occupancy"],
        type(
            "Field",
            (),
            {
                "occupancy_epistemic": inputs["components"]["occupancy_epistemic"],
                "semantic_vacuity": inputs["components"]["semantic_vacuity"],
                "total": inputs["components"]["total"],
                "lambda_occ": 1.0,
                "lambda_sem": 1.0,
            },
        )(),
        inputs["supports"],
        source_batch_size=int(args.source_batch_size),
    )
    cpu_reference_seconds = float(time.perf_counter() - cpu_started)
    gpu_reference, gpu_reference_seconds = _timed_score(
        cache, warm_rays, revision=revision, device=device
    )
    parity = {
        name: {
            "max_absolute_error": float(np.max(np.abs(gpu_reference[name] - cpu[name]))),
            "max_relative_error": float(
                np.max(
                    np.abs(gpu_reference[name] - cpu[name])
                    / np.maximum(np.abs(cpu[name]), 1.0e-8)
                )
            ),
            "allclose_rtol_1e-4_atol_1e-3": bool(
                np.allclose(
                    gpu_reference[name], cpu[name], rtol=1.0e-4, atol=1.0e-3
                )
            ),
        }
        for name in cpu
    }

    # Node-count scaling reuses the first camera and rebuilds only bounded
    # prefixes of the runtime node set.
    node_scaling = {}
    full_count = int(len(inputs["supports"]))
    node_counts = sorted(set([1, min(4, full_count), min(8, full_count), full_count]))
    for node_count in node_counts:
        local_started = time.perf_counter()
        local_cache = TorchHiddenUncertaintyCache(
            occupancy_mean=inputs["occupancy"],
            component_masses={"total": inputs["components"]["total"]},
            source_supports=inputs["supports"][:node_count],
            belief_revision=revision,
            support_provenance="runtime_cnabu_learned_component_splitter_prefix_benchmark",
            device=device,
            source_batch_size=int(args.source_batch_size),
        )
        _synchronize(device)
        build_seconds = float(time.perf_counter() - local_started)
        repetitions = []
        for _ in range(int(args.node_scaling_repeats)):
            _, elapsed = _timed_score(
                local_cache, warm_rays, revision=revision, device=device
            )
            repetitions.append(elapsed)
        node_scaling[str(node_count)] = {
            "cache_construction_seconds": build_seconds,
            "score_seconds_raw": repetitions,
            "score_seconds_median": float(np.median(repetitions)),
        }
        del local_cache
        torch.cuda.empty_cache()

    score_matrix = {
        name: np.stack(values).tolist() for name, values in score_values.items()
    }
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    view_phase = baseline["timing"]["phases"]["view_information_gain"]
    baseline_view_planning_per_call = float(
        view_phase["wall_seconds"] / view_phase["calls"]
    )
    total_score_seconds = float(sum(record["score_seconds"] for record in per_camera))
    total_ray_generation_seconds = float(
        sum(record["ray_generation_seconds"] for record in per_camera)
    )
    total_conversion_seconds = float(
        sum(record["conversion_seconds"] for record in per_camera)
    )
    total_adapter_seconds = float(
        sum(
            record["ray_generation_seconds"]
            + record["conversion_seconds"]
            + record["score_seconds"]
            for record in per_camera
        )
    )
    cached_reuse_incremental_seconds = float(
        cache_construction_seconds + total_conversion_seconds + total_score_seconds
    )
    standalone_incremental_seconds = float(
        cache_construction_seconds + total_adapter_seconds
    )
    camera_scaling_counts = sorted(
        set(
            count
            for count in (1, 4, 12, 50, 100, len(camera_indices))
            if count <= len(camera_indices)
        )
    )
    report = {
        "schema": "cnabu_hidden_uncertainty_runtime_benchmark_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "command": " ".join(sys.argv),
        "repositories": {
            "manipulation_enhanced_map_prediction": _git_state(REPO_ROOT),
            "scene_graph_mem": _git_state(SCENE_GRAPH_ROOT),
        },
        "source_hashes": {
            "benchmark_script": sha256_file(Path(__file__).resolve()),
            "audit_camera_adapter": sha256_file(
                REPO_ROOT / "shelf_gym/scripts/audit_cnabu_occlusion_attribution.py"
            ),
            "mem_attribution_adapter": sha256_file(
                REPO_ROOT / "shelf_gym/utils/cnabu_occlusion_attribution.py"
            ),
            "belief_occlusion": sha256_file(
                SCENE_GRAPH_ROOT
                / "src/scene_graph_mem/relations/belief_occlusion.py"
            ),
            "torch_attribution": sha256_file(
                SCENE_GRAPH_ROOT
                / "src/scene_graph_mem/relations/occlusion_attribution.py"
            ),
        },
        "sample": {
            "sample_id": str(args.sample_id),
            "split_role": "development",
            "cnabu_path": str(inputs["cnabu_path"]),
            "cnabu_sha256": sha256_file(inputs["cnabu_path"]),
            "learned_nodes_path": str(inputs["node_path"]),
            "learned_nodes_sha256": sha256_file(inputs["node_path"]),
            "node_count": full_count,
            "node_voxel_counts": list(inputs["sparse"].voxel_counts),
        },
        "camera": {
            "path": str(camera_path),
            "sha256": sha256_file(camera_path),
            "executable_camera_count": int(len(info_gain.camera_matrices)),
            "measured_camera_indices": list(camera_indices),
            "ray_subsample": int(args.ray_subsample),
        },
        "cache": cache.metadata.to_dict(),
        "timing": {
            "input_load_and_support_seconds": input_load_seconds,
            "cache_construction_seconds": cache_construction_seconds,
            "all_camera_score_seconds": total_score_seconds,
            "all_camera_ray_generation_seconds": total_ray_generation_seconds,
            "all_camera_conversion_seconds": total_conversion_seconds,
            "all_camera_adapter_seconds": total_adapter_seconds,
            "cached_reuse_incremental_seconds": cached_reuse_incremental_seconds,
            "standalone_incremental_seconds": standalone_incremental_seconds,
            "per_camera_score_seconds_mean": float(
                np.mean([record["score_seconds"] for record in per_camera])
            ),
            "per_camera_score_seconds_median": float(
                np.median([record["score_seconds"] for record in per_camera])
            ),
            "cpu_vectorized_reference_one_camera_seconds": cpu_reference_seconds,
            "gpu_cached_reference_one_camera_seconds": gpu_reference_seconds,
            "camera_count_scaling": _camera_scaling(
                cumulative_score_seconds, camera_scaling_counts
            ),
            "node_count_scaling": node_scaling,
            "total_script_seconds": float(time.perf_counter() - started),
        },
        "baseline_comparison": {
            "baseline_summary_path": str(baseline_path),
            "baseline_summary_sha256": sha256_file(baseline_path),
            "baseline_view_planning_seconds_per_call": baseline_view_planning_per_call,
            "ten_percent_budget_seconds": 0.1 * baseline_view_planning_per_call,
            "all_camera_score_over_baseline_view_planning": float(
                total_score_seconds / baseline_view_planning_per_call
            ),
            "all_camera_adapter_over_baseline_view_planning": float(
                total_adapter_seconds / baseline_view_planning_per_call
            ),
            "cached_reuse_incremental_over_baseline_view_planning": float(
                cached_reuse_incremental_seconds / baseline_view_planning_per_call
            ),
            "standalone_incremental_over_baseline_view_planning": float(
                standalone_incremental_seconds / baseline_view_planning_per_call
            ),
            "planner_feasible_at_ten_percent_with_existing_ray_reuse": bool(
                cached_reuse_incremental_seconds
                <= 0.1 * baseline_view_planning_per_call
            ),
        },
        "parity": parity,
        "per_camera": per_camera,
        "scores_by_camera_source": score_matrix,
        "memory": {
            "cuda_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(torch.device(device))
            ),
            "cuda_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(torch.device(device))
            ),
            "process_max_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "safety": {
            "uses_gt": False,
            "uses_simulator_instance_ids": False,
            "uses_future_observations": False,
            "training_run": False,
            "checkpoint_written": False,
            "dataset_export_written": False,
            "physical_relation_assets_or_records_loaded": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "camera_count": len(camera_indices),
                "all_camera_score_seconds": total_score_seconds,
                "all_camera_adapter_seconds": total_adapter_seconds,
                "cached_reuse_incremental_seconds": cached_reuse_incremental_seconds,
                "planner_feasible_at_ten_percent_with_existing_ray_reuse": bool(
                    cached_reuse_incremental_seconds
                    <= 0.1 * baseline_view_planning_per_call
                ),
                "cpu_reference_seconds": cpu_reference_seconds,
                "gpu_reference_seconds": gpu_reference_seconds,
                "parity": parity,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
