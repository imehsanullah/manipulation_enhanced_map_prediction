import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "shelf_gym"
    / "utils"
    / "cnabu_occlusion_attribution.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "cnabu_occlusion_attribution", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

align_oracle_supports_to_nodes = _MODULE.align_oracle_supports_to_nodes
build_gt_object_voxel_supports = _MODULE.build_gt_object_voxel_supports
build_runtime_support_partition = _MODULE.build_runtime_support_partition
dense_supports_from_sparse_indices = _MODULE.dense_supports_from_sparse_indices
info_gain_raycast_to_canonical_zyx = _MODULE.info_gain_raycast_to_canonical_zyx
match_nodes_to_gt_objects = _MODULE.match_nodes_to_gt_objects
merge_last_valid_instance_stack = _MODULE.merge_last_valid_instance_stack


def test_info_gain_raycast_conversion_preserves_order_and_invalidates_padding_or_crop():
    raw = np.asarray(
        [
            [
                [197, 11, 2],
                [196, 12, 3],
                [0, 0, 0],
                [195, 9, 4],
                [0, 12, 3],
            ]
        ],
        dtype=np.uint8,
    )

    converted = info_gain_raycast_to_canonical_zyx(
        raw,
        grid_shape_zyx=(6, 4, 200),
        crop_rows=(10, 14),
        raw_shape_hw=(20, 200),
    )

    assert converted.tolist() == [
        [[2, 1, 3], [3, 2, 4], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    ]


def test_last_valid_instance_merge_matches_existing_mem_convention():
    stack = np.asarray(
        [
            [[-1, 4], [5, -1]],
            [[7, -1], [-1, 8]],
        ]
    )
    assert merge_last_valid_instance_stack(stack).tolist() == [[7, 4], [5, 8]]


def test_gt_support_builder_keeps_object_fixed_and_missing_mass_explicit():
    hm3d = np.zeros((4, 4, 3), dtype=np.float32)
    semantic_3d = np.full((4, 4, 3), 14, dtype=np.int64)
    semantic_2d = np.full((4, 4), 14, dtype=np.int64)
    instances = np.full((1, 4, 4), -1, dtype=np.int64)

    # Two represented objects inside the crop.
    hm3d[1, 1, 0] = 1.0
    semantic_3d[1, 1, 0] = 2
    semantic_2d[1, 1] = 2
    instances[0, 1, 1] = 10
    hm3d[2, 2, 1] = 1.0
    semantic_3d[2, 2, 1] = 3
    semantic_2d[2, 2] = 3
    instances[0, 2, 2] = 20
    # Occupied object voxel without a usable instance mask.
    hm3d[1, 3, 2] = 1.0
    semantic_3d[1, 3, 2] = 4
    semantic_2d[1, 3] = 4
    # Fixed environment.
    hm3d[2, 0, 0] = 1.0

    result = build_gt_object_voxel_supports(
        hm3d=hm3d,
        semantic_2d=semantic_2d,
        semantic_3d=semantic_3d,
        instance_maps=instances,
        crop_rows=(1, 3),
    )

    assert result.instance_ids == (10, 20)
    assert result.class_ids == (2, 3)
    assert result.supports_zyx.shape == (2, 3, 2, 4)
    assert result.supports_zyx[0, 0, 0, 1]
    assert result.supports_zyx[1, 1, 1, 2]
    assert result.unrepresented_object_support_zyx[2, 0, 3]
    assert result.fixed_environment_support_zyx[0, 1, 0]
    assert result.coverage_summary()["object_support_coverage"] == pytest.approx(2 / 3)


def test_class_aware_hungarian_matching_and_oracle_alignment_report_unmatched_gt():
    gt_masks = np.zeros((3, 4, 5), dtype=bool)
    gt_masks[0, :2, :2] = True
    gt_masks[1, 2:, :2] = True
    gt_masks[2, :, 3:] = True
    node_masks = np.stack([gt_masks[1], gt_masks[0]])
    matching = match_nodes_to_gt_objects(
        node_masks_raw_hw=node_masks,
        node_class_ids=[2, 1],
        gt_masks_raw_hw=gt_masks,
        gt_class_ids=[1, 2, 3],
        iou_threshold=0.25,
    )

    assert matching.node_to_gt_index == (1, 0)
    assert matching.matched_iou == (1.0, 1.0)
    assert matching.unmatched_gt_indices == (2,)

    supports = np.zeros((3, 2, 2, 3), dtype=bool)
    supports[0, 0, 0, 0] = True
    supports[1, 0, 1, 0] = True
    supports[2, 1, 1, 2] = True
    gt = _MODULE.GTObjectVoxelSupports(
        supports_zyx=supports,
        masks_raw_hw=gt_masks,
        instance_ids=(10, 20, 30),
        class_ids=(1, 2, 3),
        fixed_environment_support_zyx=np.zeros((2, 2, 3), dtype=bool),
        unrepresented_object_support_zyx=np.zeros((2, 2, 3), dtype=bool),
        occupied_support_zyx=supports.any(axis=0),
        crop_rows=(1, 3),
        occupancy_threshold=0.5,
    )
    aligned = align_oracle_supports_to_nodes(gt, matching)

    assert np.array_equal(aligned.source_supports_zyx[0], supports[1])
    assert np.array_equal(aligned.source_supports_zyx[1], supports[0])
    assert aligned.target_source_indices.tolist() == [0, 1]
    assert aligned.unrepresented_support_zyx[1, 1, 2]


def test_runtime_partition_is_gt_free_and_exhaustive_over_occupied_belief():
    occupancy = np.zeros((2, 3, 4), dtype=np.float32)
    occupancy[0, 0, 0] = 0.9
    occupancy[0, 1, 1] = 0.9
    occupancy[1, 2, 2] = 0.9
    semantic = np.zeros((15, 3, 4), dtype=np.float32)
    semantic[1] = 1.0
    semantic[:, 2, 2] = 0.0
    semantic[14, 2, 2] = 1.0
    sources = dense_supports_from_sparse_indices(
        [np.asarray([[0, 0, 0]])], grid_shape_zyx=occupancy.shape
    )

    partition = build_runtime_support_partition(
        occupancy_mean=occupancy,
        semantic_mean=semantic,
        source_supports_zyx=sources,
    )

    assert partition.unrepresented_support_zyx[0, 1, 1]
    assert partition.fixed_environment_support_zyx[1, 2, 2]
    assert partition.coverage_summary()["partition_coverage"] == pytest.approx(1.0)
