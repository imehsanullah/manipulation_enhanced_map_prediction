# Existing Datasets on Wonka

This note catalogs the visible contents of `/home/data` on `wonka` and marks
which entries appear related to `manipulation_enhanced_map_prediction`.

Checked on 2026-05-01. No `/home/data` files were modified.

## Directly Related

These match the map-completion / push-completion naming and data schemas used by
this project.

### Best Existing Packed Training Files

These are the most useful existing files for original paper-style supervised
training:

```text
/home/data/map_completion_30k.hdf5
/home/data/push_completion_10k.hdf5
```

Observed contents:

- `/home/data/map_completion_30k.hdf5`
  - Size: about `649G`
  - Samples: `30047`
  - Keys include `hms`, `semantic_hms`, `depths`, `gt_2d`, `gt_3d`,
    `gt_semantics`, `gt_semantics_3d`, `semantics`
  - This is very close to the paper-scale map-completion dataset.
- `/home/data/push_completion_10k.hdf5`
  - Size: about `252G`
  - Samples: `11646`
  - Keys include `hms`, `semantic_hms`, `depths`, `gt_2d`, `gt_3d`,
    `gt_semantics`, `post_push_gt_3d`, `post_push_gt_semantics`,
    `differences`, `swept_volume`, `push_parametrization`, `semantics`
  - This is very close to the paper-scale push-completion dataset.

Important: these existing datasets do **not** include the new `instance_maps`
fields we added locally. They are useful for the original training setup, but
not for training or analysis that requires per-object instance masks.

### Raw Map-Completion Data

These directories use the raw generated layout with job-id/sample-id folders and
files like `pre_action/hms.npz`, `pre_action/gt_hms.npz`,
`pre_action/camera_matrices.npz`, and `pre_action/placed_objects.pkl`.

```text
/home/data/map_completion_30k
/home/data/map_completion_ycb_new_depth_300
/home/data/map_completion_ycb_new_depth_5k
/home/data/data_matti/map_data
```

Example observed raw map sample:

```text
/home/data/map_completion_30k/15/000001457/pre_action/hms.npz
/home/data/map_completion_30k/15/000001457/pre_action/gt_hms.npz
/home/data/map_completion_30k/15/000001457/pre_action/camera_matrices.npz
/home/data/map_completion_30k/15/000001457/pre_action/placed_objects.pkl
```

The inspected raw `.npz` files contain original map fields such as `hms`,
`dilated_hms`, `semantic_hms`, `semantics`, `depths`, `gt_hms`, `hm3d`,
`semantic_2d`, and `semantic_3d`. They did not contain `instance_maps`.

### Packed Map-Completion HDF5 Files

These are related packed HDF5 datasets:

```text
/home/data/map_completion_30k.hdf5
/home/data/map_completion_fixed.hdf5
/home/data/map_completion_matti_large.hdf5
/home/data/h5py_data/large_map_completion_fixed.hdf5
/home/data/h5py_data/map_completion_ycb_full.hdf5
/home/data/h5py_data/map_completion_ycb_new_depth.hdf5
/home/data/h5py_data/map_completion_ycb_new_depth_300.hdf5
/home/data/h5py_data/map_completion_ycb_new_depth_5k.hdf5
/home/data/h5py_data/mapping_5k.hdf5
/home/data/h5py_data/mapping_5k_fix.hdf5
/home/data/split_files/mapping_5k.hdf5
```

Observed sample counts and notes:

- `/home/data/map_completion_30k.hdf5`: `30047` samples.
- `/home/data/map_completion_fixed.hdf5`: `100032` samples.
- `/home/data/map_completion_matti_large.hdf5`: `30012` samples, includes a
  `cameras` group.
- `/home/data/h5py_data/map_completion_ycb_full.hdf5`: `28800` samples.
- `/home/data/h5py_data/map_completion_ycb_new_depth.hdf5`: `4243` samples.
- `/home/data/h5py_data/map_completion_ycb_new_depth_300.hdf5`: `19844`
  samples.
- `/home/data/h5py_data/map_completion_ycb_new_depth_5k.hdf5`: `9537` samples.
- `/home/data/split_files/mapping_5k.hdf5`: appeared truncated when opened with
  `h5py`, so do not use it without repair or verification.

### Raw Push-Completion Data

These directories use the raw generated push layout with `pre_action`,
`post_action`, and `swept_volume` folders:

```text
/home/data/push_completion
/home/data/push_completion_30k
/home/data/push_completion_ycb
/home/data/push_completion_ycb_90deg
/home/data/push_completion_ycb_new_depth
/home/data/push_completion_ycb_new_depth_300
/home/data/push_completion_ycb_new_depth_5k
/home/data/push_completion_ycb_new_depth_5k_obj_fix
```

Example observed raw push sample:

```text
/home/data/push_completion_30k/15/000000804/pre_action/hms.npz
/home/data/push_completion_30k/15/000000804/pre_action/gt_hms.npz
/home/data/push_completion_30k/15/000000804/post_action/hms.npz
/home/data/push_completion_30k/15/000000804/post_action/gt_hms.npz
/home/data/push_completion_30k/15/000000804/swept_volume/swept_map.npz
```

The inspected `swept_map.npz` contains `swept_map` and
`motion_parametrization`.

### Packed Push HDF5 Files

These are related packed HDF5 datasets:

```text
/home/data/push_completion_10k.hdf5
/home/data/push_data_matti.hdf5
/home/data/h5py_data/large_push_prediction.hdf5
/home/data/h5py_data/push_completion_ycb_90deg.hdf5
/home/data/h5py_data/push_completion_ycb_depth_full.hdf5
/home/data/h5py_data/push_completion_ycb_depth_full_canceled.hdf5
/home/data/h5py_data/push_completion_ycb_new_depth_5k.hdf5
/home/data/h5py_data/push_completion_ycb_new_depth_5k_obj_fix.hdf5
/home/data/h5py_data/push_prediction.hdf5
/home/data/h5py_data/push_prediction_ycb.hdf5
```

Observed sample counts and notes:

- `/home/data/push_completion_10k.hdf5`: `11646` samples.
- `/home/data/push_data_matti.hdf5`: `5007` samples; inspected
  `swept_volume` shape was `(5007, 102, 120, 200)`, which differs from the newer
  `(N, 102, 140, 200)` format.
- `/home/data/h5py_data/push_completion_ycb_new_depth_5k.hdf5`: `5862`
  samples.
- `/home/data/h5py_data/push_completion_ycb_new_depth_5k_obj_fix.hdf5`: `4922`
  samples.
- `/home/data/h5py_data/push_prediction_ycb.hdf5`: `10353` samples.
- `/home/data/h5py_data/large_push_prediction.hdf5`: `9774` samples.
- `/home/data/h5py_data/push_completion_ycb_90deg.hdf5`: `4838` samples.

### Related Code, Logs, or Split Artifacts

These are related to `shelf_gym` / manipulation mapping, but are not the main
packed supervised datasets:

```text
/home/data/train_vpp_push
/home/data/Mati_ICRA
/home/data/split_files
/home/data/split_files_pushing
```

Notes:

- `/home/data/train_vpp_push` contains `train.py`, `train_push.py`, W&B runs,
  and TensorBoard logs for VPP/RL-style push training.
- `/home/data/Mati_ICRA` contains an old `pybullet_shelf_gym` folder.
- `/home/data/split_files` and `/home/data/split_files_pushing` look like split
  or chunk artifacts related to map/push datasets. Verify individual files
  before using them.

## Probably Unrelated Or Other Projects

These do not appear to be the map/push completion datasets for this project.

### Grasping Datasets And Outputs

```text
/home/data/grasp_completion
/home/data/grasp_completion_2
/home/data/grasp_full_output
/home/data/grasp_full_output_fixed
/home/data/grasping_5k.hdf5
/home/data/grasping_5k_2.hdf5
/home/data/grasp_output_7k.hdf5
/home/data/new_grasping_5k
/home/data/new_grasping_5k.hdf5
/home/data/new_grasping_5k_full.hdf5
/home/data/new_grasping_5k_full_2.0.hdf5
```

These are likely for grasp prediction/completion rather than
manipulation-enhanced map prediction.

### Other Project Directories

```text
/home/data/DRL-robot-navigation
/home/data/menon
/home/data/sicong
```

Notes:

- `/home/data/DRL-robot-navigation` appears to be a separate navigation project.
- `/home/data/menon` contains other research/project folders such as
  `data_pepper_scenes`, `experiments`, and `menon26icra`.
- `/home/data/sicong` contains `network_training_data` scripts and many
  normalized objects. It does not match the MEM map/push dataset schema.

## Practical Recommendation

For original MEM training, use symlinks to these existing packed datasets rather
than regenerating them:

```bash
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction/shelf_gym/scripts/model_training

ln -sf /home/data/map_completion_30k.hdf5 map_completion_fine_tune.hdf5
ln -sf /home/data/push_completion_10k.hdf5 unbiased_push_dataset.hdf5
```

Do not use these existing packed files if the run needs the new per-object
instance-mask fields. For instance masks, use the newly patched local generation
and packing path under `/tmp/manipulation_map_data`.
