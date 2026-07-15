#!/usr/bin/env python3
"""Inspect or export the v1 carried-geometry action-conditioned oracle.

The default invocation is read-only: it replays scenes in PyBullet and prints
summaries.  Supplying ``--output-dir`` exports the full evidence records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from shelf_gym.environments.shelf_environment import ShelfEnv
from shelf_gym.utils.action_conditioned_relation_oracle import (
    OracleActionFamilyConfig,
    aggregate_prototype_records,
    evaluate_saved_scene,
    render_scene_oracle_debug,
)


DEFAULT_DATA_ROOT = Path("/data/manipulation_map_data/raw/map_data")
DEFAULT_RECORDS_JSON = Path(
    "/data/manipulation_map_data/derived/cnabu_scene_graph/"
    "learned_splitter_nodes_1000_20260712/records_with_learned_nodes.json"
)


def _load_manifest_records(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("records manifest must contain a JSON list")
    return records


def select_scene_disjoint_round_robin(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    start_index: int = 0,
) -> List[Path]:
    """Select across top-level scene groups before taking a second sample."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    grouped: Dict[str, deque[Path]] = defaultdict(deque)
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        sample_dir = record.get("sample_dir")
        if not sample_id or not sample_dir:
            continue
        grouped[sample_id.split("/", 1)[0]].append(Path(str(sample_dir)))
    if not grouped:
        raise ValueError("records manifest contains no sample_id/sample_dir pairs")

    group_names = sorted(grouped, key=lambda value: (int(value) if value.isdigit() else sys.maxsize, value))
    ordered: List[Path] = []
    while any(grouped.values()):
        for name in group_names:
            if grouped[name]:
                ordered.append(grouped[name].popleft())
    return ordered[int(start_index) : int(start_index) + int(limit)]


def _resolve_sample_dirs(args: argparse.Namespace) -> List[Path]:
    if args.sample_dir:
        return [Path(item).resolve() for item in args.sample_dir]
    records = _load_manifest_records(args.records_json)
    return select_scene_disjoint_round_robin(records, limit=args.limit, start_index=args.start_index)


def _compact_scene_result(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        **record["scene_summary"],
        "geometry_edge_count": record["geometry_pseudo_gt_v0"]["edge_count"],
        "geometry_vs_action": record["geometry_vs_action_comparison"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", action="append", help="Explicit pre_action directory; repeatable")
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS_JSON)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Export JSON records here. If omitted, the command is strictly no-write.",
    )
    parser.add_argument(
        "--render-debug",
        action="store_true",
        help="Write one 3D trajectory diagnostic per scene (requires --output-dir).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_debug and args.output_dir is None:
        raise SystemExit("--render-debug requires --output-dir")
    if args.output_dir is not None and args.output_dir.exists():
        raise SystemExit("refusing to overwrite existing output directory: {}".format(args.output_dir))

    sample_dirs = _resolve_sample_dirs(args)
    missing = [str(path) for path in sample_dirs if not path.is_dir()]
    if missing:
        raise SystemExit("missing pre_action directories: {}".format(missing))
    if not sample_dirs:
        raise SystemExit("no scenes selected")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True)

    records: List[Dict[str, Any]] = []
    environment = ShelfEnv(render=False, max_obj_num=25, use_ycb=True)
    try:
        for index, sample_dir in enumerate(sample_dirs):
            record, _ = evaluate_saved_scene(
                environment,
                pre_action_dir=sample_dir,
                config=OracleActionFamilyConfig(),
            )
            records.append(record)
            print(json.dumps(_compact_scene_result(record), sort_keys=True), flush=True)
            if output_dir is not None:
                _write_json(output_dir / "scene_{:03d}.json".format(index), record)
                if args.render_debug:
                    render_scene_oracle_debug(record, output_dir / "scene_{:03d}.png".format(index))
    finally:
        environment.close()

    summary = aggregate_prototype_records(records)
    summary["selected_sample_dirs"] = [str(path) for path in sample_dirs]
    summary["mode"] = "export" if output_dir is not None else "read_only_no_write"
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if output_dir is not None:
        _write_json(output_dir / "summary.json", summary)
        _write_json(
            output_dir / "manifest.json",
            {
                "schema": "action_conditioned_oracle_prototype_manifest_v1",
                "relation_target_method": summary["relation_target_method"],
                "scene_files": ["scene_{:03d}.json".format(index) for index in range(len(records))],
                "summary_file": "summary.json",
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
