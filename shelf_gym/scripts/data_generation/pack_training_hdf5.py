import argparse
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm


CAMERA_KEYS = (
    "position",
    "pybullet_extrinsic_matrix",
    "projection_matrix",
    "intrinsic_matrix",
    "o3d_extrinsic_matrix",
)


def compression_kwargs(enabled=True):
    if not enabled:
        return {}
    return hdf5plugin.Blosc(cname="zstd", clevel=3, shuffle=hdf5plugin.Blosc.BITSHUFFLE)


def discover_map_samples(root):
    root = Path(root)
    samples = []
    for pre_action in sorted(root.glob("*/*/pre_action")):
        if (pre_action / "hms.npz").exists() and (pre_action / "gt_hms.npz").exists():
            samples.append(pre_action)
    return samples


def discover_push_samples(root):
    root = Path(root)
    samples = []
    for sample_dir in sorted(root.glob("*/*")):
        required = [
            sample_dir / "pre_action" / "hms.npz",
            sample_dir / "pre_action" / "gt_hms.npz",
            sample_dir / "post_action" / "gt_hms.npz",
            sample_dir / "swept_volume" / "swept_map.npz",
        ]
        if all(path.exists() for path in required):
            samples.append(sample_dir)
    return samples


def load_observation_npz(path):
    with np.load(path, allow_pickle=True) as data:
        hms = data["hms"].astype(np.float16)
        semantic_hms = data["semantic_hms"].astype(np.uint8)
        instance_maps = data["instance_maps"].astype(np.int32) if "instance_maps" in data else None
        depths = data["depths"].astype(np.uint16)
    return hms, semantic_hms, instance_maps, depths


def load_gt_npz(path):
    with np.load(path, allow_pickle=True) as data:
        gt_2d = data["gt_hms"].astype(np.float16)
        gt_3d = np.moveaxis(data["hm3d"], 2, 0).astype(bool)
        gt_semantics = data["semantic_2d"].astype(np.uint8)
        gt_semantics_3d = np.moveaxis(data["semantic_3d"], 2, 0).astype(np.uint8)
        gt_instance_maps = data["instance_maps"].astype(np.int32) if "instance_maps" in data else None
    return gt_2d, gt_3d, gt_semantics, gt_semantics_3d, gt_instance_maps


def load_camera_npz(path):
    with np.load(path, allow_pickle=True) as data:
        cameras = data["obj_ids"]
    stacked = {}
    for key in CAMERA_KEYS:
        stacked[key] = np.stack([np.asarray(camera[key]) for camera in cameras])
    return stacked


def create_dataset(group, name, shape, dtype, chunks, compress):
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        **compression_kwargs(compress),
    )


def create_camera_group(handle, n_samples, first_camera_data, compress):
    group = handle.create_group("cameras")
    datasets = {}
    for key, value in first_camera_data.items():
        shape = (n_samples,) + value.shape
        chunks = (1,) + value.shape
        datasets[key] = create_dataset(group, key, shape, value.dtype, chunks, compress)
    return datasets


def write_map_hdf5(samples, output, compress=True):
    if not samples:
        raise RuntimeError("No map samples found")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    first_hms, first_semantic_hms, first_instance_maps, first_depths = load_observation_npz(samples[0] / "hms.npz")
    first_gt_2d, first_gt_3d, first_gt_semantics, first_gt_semantics_3d, first_gt_instance_maps = load_gt_npz(samples[0] / "gt_hms.npz")
    first_camera_data = load_camera_npz(samples[0] / "camera_matrices.npz")

    n = len(samples)
    with h5py.File(output, "w") as handle:
        handle.attrs["source"] = "raw map_collection.py output"
        handle.attrs["num_samples"] = n
        hms = create_dataset(handle, "hms", (n,) + first_hms.shape, first_hms.dtype, (1, 10) + first_hms.shape[1:], compress)
        semantic_hms = create_dataset(handle, "semantic_hms", (n,) + first_semantic_hms.shape, first_semantic_hms.dtype, (1, 10) + first_semantic_hms.shape[1:], compress)
        instance_hms = None
        if first_instance_maps is not None:
            instance_hms = create_dataset(handle, "instance_hms", (n,) + first_instance_maps.shape, first_instance_maps.dtype, (1, 10) + first_instance_maps.shape[1:], compress)
        depths = create_dataset(handle, "depths", (n,) + first_depths.shape, first_depths.dtype, (1, 10) + first_depths.shape[1:], compress)
        gt_2d = create_dataset(handle, "gt_2d", (n,) + first_gt_2d.shape, first_gt_2d.dtype, (1,) + first_gt_2d.shape, compress)
        gt_3d = create_dataset(handle, "gt_3d", (n,) + first_gt_3d.shape, first_gt_3d.dtype, (1,) + first_gt_3d.shape, compress)
        gt_semantics = create_dataset(handle, "gt_semantics", (n,) + first_gt_semantics.shape, first_gt_semantics.dtype, (1,) + first_gt_semantics.shape, compress)
        gt_semantics_3d = create_dataset(handle, "gt_semantics_3d", (n,) + first_gt_semantics_3d.shape, first_gt_semantics_3d.dtype, (1,) + first_gt_semantics_3d.shape, compress)
        gt_instance_maps = None
        if first_gt_instance_maps is not None:
            gt_instance_maps = create_dataset(handle, "gt_instance_maps", (n,) + first_gt_instance_maps.shape, first_gt_instance_maps.dtype, (1,) + first_gt_instance_maps.shape, compress)
        cameras = create_camera_group(handle, n, first_camera_data, compress)

        for idx, sample in enumerate(tqdm(samples, desc="packing map hdf5")):
            obs_hms, obs_semantics, obs_instances, obs_depths = load_observation_npz(sample / "hms.npz")
            sample_gt_2d, sample_gt_3d, sample_gt_semantics, sample_gt_semantics_3d, sample_gt_instances = load_gt_npz(sample / "gt_hms.npz")
            camera_data = load_camera_npz(sample / "camera_matrices.npz")
            hms[idx] = obs_hms
            semantic_hms[idx] = obs_semantics
            if instance_hms is not None:
                instance_hms[idx] = obs_instances
            depths[idx] = obs_depths
            gt_2d[idx] = sample_gt_2d
            gt_3d[idx] = sample_gt_3d
            gt_semantics[idx] = sample_gt_semantics
            gt_semantics_3d[idx] = sample_gt_semantics_3d
            if gt_instance_maps is not None:
                gt_instance_maps[idx] = sample_gt_instances
            for key, dataset in cameras.items():
                dataset[idx] = camera_data[key]


def normalize_swept_volume(swept_volume):
    swept_volume = swept_volume.astype(bool)
    if swept_volume.shape == (102, 120, 200):
        return np.pad(swept_volume, ((0, 0), (10, 10), (0, 0)), mode="constant")
    if swept_volume.shape != (102, 140, 200):
        raise ValueError(f"Unexpected swept volume shape {swept_volume.shape}")
    return swept_volume


def load_swept_npz(path):
    with np.load(path, allow_pickle=True) as data:
        swept = normalize_swept_volume(data["swept_map"])
        push_parametrization = data["motion_parametrization"].astype(np.int16)
    if push_parametrization.shape != (6,):
        push_parametrization = push_parametrization.reshape(-1)[:6].astype(np.int16)
    return swept, push_parametrization


def write_push_hdf5(samples, output, compress=True):
    if not samples:
        raise RuntimeError("No push samples found")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    first_hms, first_semantic_hms, first_instance_maps, first_depths = load_observation_npz(samples[0] / "pre_action" / "hms.npz")
    first_gt_2d, first_gt_3d, first_gt_semantics, _, first_gt_instance_maps = load_gt_npz(samples[0] / "pre_action" / "gt_hms.npz")
    _, first_post_gt_3d, first_post_gt_semantics, _, first_post_gt_instance_maps = load_gt_npz(samples[0] / "post_action" / "gt_hms.npz")
    first_swept, first_push = load_swept_npz(samples[0] / "swept_volume" / "swept_map.npz")
    first_difference = first_gt_semantics != first_post_gt_semantics

    n = len(samples)
    with h5py.File(output, "w") as handle:
        handle.attrs["source"] = "raw pushing_collection.py output"
        handle.attrs["num_samples"] = n
        hms = create_dataset(handle, "hms", (n,) + first_hms.shape, first_hms.dtype, (1, 10) + first_hms.shape[1:], compress)
        semantic_hms = create_dataset(handle, "semantic_hms", (n,) + first_semantic_hms.shape, first_semantic_hms.dtype, (1, 10) + first_semantic_hms.shape[1:], compress)
        instance_hms = None
        if first_instance_maps is not None:
            instance_hms = create_dataset(handle, "instance_hms", (n,) + first_instance_maps.shape, first_instance_maps.dtype, (1, 10) + first_instance_maps.shape[1:], compress)
        depths = create_dataset(handle, "depths", (n,) + first_depths.shape, first_depths.dtype, (1, 10) + first_depths.shape[1:], compress)
        gt_2d = create_dataset(handle, "gt_2d", (n,) + first_gt_2d.shape, first_gt_2d.dtype, (1,) + first_gt_2d.shape, compress)
        gt_3d = create_dataset(handle, "gt_3d", (n,) + first_gt_3d.shape, first_gt_3d.dtype, (1,) + first_gt_3d.shape, compress)
        gt_semantics = create_dataset(handle, "gt_semantics", (n,) + first_gt_semantics.shape, first_gt_semantics.dtype, (1,) + first_gt_semantics.shape, compress)
        gt_instance_maps = None
        if first_gt_instance_maps is not None:
            gt_instance_maps = create_dataset(handle, "gt_instance_maps", (n,) + first_gt_instance_maps.shape, first_gt_instance_maps.dtype, (1,) + first_gt_instance_maps.shape, compress)
        post_push_gt_3d = create_dataset(handle, "post_push_gt_3d", (n,) + first_post_gt_3d.shape, first_post_gt_3d.dtype, (1,) + first_post_gt_3d.shape, compress)
        post_push_gt_semantics = create_dataset(handle, "post_push_gt_semantics", (n,) + first_post_gt_semantics.shape, first_post_gt_semantics.dtype, (1,) + first_post_gt_semantics.shape, compress)
        post_push_gt_instance_maps = None
        if first_post_gt_instance_maps is not None:
            post_push_gt_instance_maps = create_dataset(handle, "post_push_gt_instance_maps", (n,) + first_post_gt_instance_maps.shape, first_post_gt_instance_maps.dtype, (1,) + first_post_gt_instance_maps.shape, compress)
        differences = create_dataset(handle, "differences", (n,) + first_difference.shape, first_difference.dtype, (1,) + first_difference.shape, compress)
        swept_volume = create_dataset(handle, "swept_volume", (n,) + first_swept.shape, first_swept.dtype, (1,) + first_swept.shape, compress)
        push_parametrization = create_dataset(handle, "push_parametrization", (n,) + first_push.shape, first_push.dtype, (1,) + first_push.shape, compress)

        for idx, sample in enumerate(tqdm(samples, desc="packing push hdf5")):
            obs_hms, obs_semantics, obs_instances, obs_depths = load_observation_npz(sample / "pre_action" / "hms.npz")
            sample_gt_2d, sample_gt_3d, sample_gt_semantics, _, sample_gt_instances = load_gt_npz(sample / "pre_action" / "gt_hms.npz")
            _, sample_post_gt_3d, sample_post_gt_semantics, _, sample_post_gt_instances = load_gt_npz(sample / "post_action" / "gt_hms.npz")
            sample_swept, sample_push = load_swept_npz(sample / "swept_volume" / "swept_map.npz")
            hms[idx] = obs_hms
            semantic_hms[idx] = obs_semantics
            if instance_hms is not None:
                instance_hms[idx] = obs_instances
            depths[idx] = obs_depths
            gt_2d[idx] = sample_gt_2d
            gt_3d[idx] = sample_gt_3d
            gt_semantics[idx] = sample_gt_semantics
            if gt_instance_maps is not None:
                gt_instance_maps[idx] = sample_gt_instances
            post_push_gt_3d[idx] = sample_post_gt_3d
            post_push_gt_semantics[idx] = sample_post_gt_semantics
            if post_push_gt_instance_maps is not None:
                post_push_gt_instance_maps[idx] = sample_post_gt_instances
            differences[idx] = sample_gt_semantics != sample_post_gt_semantics
            swept_volume[idx] = sample_swept
            push_parametrization[idx] = sample_push


def apply_limit(samples, limit):
    if limit is None:
        return samples
    return samples[:limit]


def parse_args():
    parser = argparse.ArgumentParser(description="Pack raw MEM data-generation folders into training HDF5 files.")
    parser.add_argument("--map-root", default="/tmp/manipulation_map_data/raw/map_data")
    parser.add_argument("--push-root", default="/tmp/manipulation_map_data/raw/push_data")
    parser.add_argument("--map-output", default="/tmp/manipulation_map_data/model_training/map_completion_fine_tune.hdf5")
    parser.add_argument("--push-output", default="/tmp/manipulation_map_data/model_training/unbiased_push_dataset.hdf5")
    parser.add_argument("--kind", choices=["map", "push", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-compression", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    compress = not args.no_compression
    if args.kind in ("map", "both"):
        map_samples = apply_limit(discover_map_samples(args.map_root), args.limit)
        print(f"Found {len(map_samples)} map samples under {args.map_root}")
        write_map_hdf5(map_samples, args.map_output, compress=compress)
        print(f"Wrote {args.map_output}")
    if args.kind in ("push", "both"):
        push_samples = apply_limit(discover_push_samples(args.push_root), args.limit)
        print(f"Found {len(push_samples)} push samples under {args.push_root}")
        write_push_hdf5(push_samples, args.push_output, compress=compress)
        print(f"Wrote {args.push_output}")


if __name__ == "__main__":
    main()
