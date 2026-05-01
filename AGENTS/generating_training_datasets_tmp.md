# Generating MEM Training Datasets on `/tmp`

This documents the local dataset setup for `manipulation_enhanced_map_prediction`.
The repository partition under `/home/user` is nearly full, so generated data is
kept under `/tmp` and symlinked back into the checkout.

## Storage Layout

```text
/tmp/manipulation_map_data/
  raw/
    map_data/
    push_data/
  model_training/
  logs/
```

Repo symlinks:

```text
shelf_gym/data -> /tmp/manipulation_map_data/raw
shelf_gym/scripts/model_training/map_completion_fine_tune.hdf5
  -> /tmp/manipulation_map_data/model_training/map_completion_fine_tune.hdf5
shelf_gym/scripts/model_training/unbiased_push_dataset.hdf5
  -> /tmp/manipulation_map_data/model_training/unbiased_push_dataset.hdf5
```

The default data-generation config paths still use `../../data/...`, which now
resolve through `shelf_gym/data` into `/tmp`.

## Environment Fixes Applied

The `manipulation_map` environment originally had:

```text
numpy 2.4.3
shapely 2.0.4
```

That combination failed even basic `MultiPoint` construction, blocking object
placement. Shapely was upgraded to:

```text
shapely 2.1.2
```

The repo metadata was updated from `shapely==2.0.4` to `shapely>=2.1.0` in:

```text
requirements.txt
setup.py
```

## Raw Data Generation

Run from:

```bash
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction/shelf_gym/scripts/data_generation
```

Map-only data:

```bash
/home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python map_collection.py
```

Push data:

```bash
/home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python pushing_collection.py
```

The config files are:

```text
config/map_collection_config.yaml
config/pushing_collection_config.yaml
```

Important notes:

- `pushing_collection.py` was fixed to read `pushing_collection_config.yaml` by
  default.
- `pushing_collection_config.yaml` now uses `use_ycb: True`, because the
  training code expects 15 semantic classes.
- `max_dataset_size` is inclusive in the current loop. A value of `999` produces
  about 1000 samples for a single job.
- Start with one process. Parallel generation uses more EGL/OpenGL contexts,
  more CPU, and much more disk I/O.

## Running Generation With tmux

Use `tmux` for long-running generation jobs so they survive terminal
disconnects. The current single-session command runs map generation first and
then push generation, using GPU 0 for CUDA/CuPy-visible work and writing a log
under `/tmp/manipulation_map_data/logs`. The script entrypoints force GUI
rendering in their default single-worker mode, so the command calls `run(True,
...)` directly to use the existing headless path without launching many
parallel workers.

```bash
tmux new-session -d -s mem_data_generation \
  "bash -lc 'set -eo pipefail; \
   cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction/shelf_gym/scripts/data_generation; \
   LOG=/tmp/manipulation_map_data/logs/data_generation_\$(date +%Y%m%d_%H%M%S).log; \
   echo \"log: \$LOG\" | tee -a \"\$LOG\"; \
   echo \"starting map generation: \$(date)\" | tee -a \"\$LOG\"; \
   CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python -c \"from map_collection import parse_config, run; run(True, parse_config(), 0)\" 2>&1 | tee -a \"\$LOG\"; \
   echo \"starting push generation: \$(date)\" | tee -a \"\$LOG\"; \
   CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python -c \"from pushing_collection import parse_config, run; run(True, parse_config(), 0)\" 2>&1 | tee -a \"\$LOG\"; \
   echo \"finished: \$(date)\" | tee -a \"\$LOG\"'"
```

Monitor:

```bash
tmux attach -t mem_data_generation
```

Detach without stopping it:

```text
Ctrl-b then d
```

Find the current log:

```bash
ls -t /tmp/manipulation_map_data/logs/data_generation_*.log | head -1
```

Tail the log without attaching:

```bash
tail -f "$(ls -t /tmp/manipulation_map_data/logs/data_generation_*.log | head -1)"
```

Stop the job:

```bash
tmux kill-session -t mem_data_generation
```

## HDF5 Packing

Raw generation writes per-sample `.npz` folders. Training expects HDF5 files.
Use:

```bash
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction

/home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python \
  shelf_gym/scripts/data_generation/pack_training_hdf5.py \
  --kind map \
  --map-root /tmp/manipulation_map_data/raw/map_data \
  --map-output /tmp/manipulation_map_data/model_training/map_completion_fine_tune.hdf5

/home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python \
  shelf_gym/scripts/data_generation/pack_training_hdf5.py \
  --kind push \
  --push-root /tmp/manipulation_map_data/raw/push_data \
  --push-output /tmp/manipulation_map_data/model_training/unbiased_push_dataset.hdf5
```

For a quick test, pass `--limit N`.

Instance maps are preserved for analysis/visualization:

```text
hms.npz:
  instance_maps                  # observed per-camera instance-id maps

gt_hms.npz:
  instance_maps                  # privileged/top-down ground-truth instance-id maps

packed map HDF5:
  instance_hms
  gt_instance_maps

packed push HDF5:
  instance_hms
  gt_instance_maps
  post_push_gt_instance_maps
```

These are instance-id maps. Convert each unique non-background instance id into
a binary mask when per-object masks are needed.

## Smoke Test Results

Map generation smoke:

```text
/tmp/manipulation_map_data/raw/map_data_smoke/1/000000000/pre_action/
  camera_matrices.npz
  gt_hms.npz
  hms.npz
  placed_objects.pkl
```

One map scene took about 69 seconds and wrote an `hms.npz` of about 17 MB.

Push generation smoke:

```text
/tmp/manipulation_map_data/raw/push_data_smoke_fixed/0/000000000/
  pre_action/
  swept_volume/
  post_action/
```

One push scene took about 101 seconds. The fixed raw push sample contains:

```text
swept_map shape: (102, 140, 200)
motion_parametrization shape: (6,)
```

Packed smoke HDF5 files were written to:

```text
/tmp/manipulation_map_data/model_training/map_completion_smoke.hdf5
/tmp/manipulation_map_data/model_training/unbiased_push_smoke.hdf5
```

Both packed files were validated through the project dataset classes.

An additional instance-map smoke sample was generated at:

```text
/tmp/manipulation_map_data/raw/map_data_instance_smoke
```

It confirmed:

```text
hms.npz instance_maps:        (300, 140, 200), int32 after packing
gt_hms.npz instance_maps:     (6, 140, 200), int32 after packing
HDF5 instance_hms:            (1, 300, 140, 200)
HDF5 gt_instance_maps:        (1, 6, 140, 200)
```

## Paper-Scale Targets

The paper reports:

```text
30,000 map-completion scenes
11,700 push samples
train/val/test split: 0.8 / 0.1 / 0.1
```

This will likely consume large disk space. Raw `.npz` files are compressed, but
each scene still contains 300 camera views and depth maps. The HDF5 packer uses
Blosc/Zstd compression.
