# MEM reproduction

This branch starts at official MEM main `8ffa53f0c32e31f81ac9c1a75612b2e5a9443456`.
It is separate from the scene-graph `dev` branch. It retains the official model
architecture, semantic-distance push sampling and ranking weight `w_conf=2.0`.

## Corrections

- Execute observations even when no push was selected and there was no recent push.
- Retain the push-selected viewpoint for the observation following a successful push.
- Treat absent motion parameters as no feasible push candidates.
- Log and skip candidates whose construction raises Shapely GEOSException; unrelated errors propagate.
- Discover shelf_gym packages explicitly during installation.

The first two restore behavior present in the recovered legacy implementation.
The other changes support execution. This is corrected official code, not an
untouched release or the exact legacy source.

## Evaluator provenance

`mem_evaluator.py` derives from recovered `eval_rss_method.py`, from Nils's
archive `projects_20260708.zip`, under `projects/cleaned/pybullet_shelf_gym`.
The preserved successful reconstruction is documented in thesis_records under
`diagnostics/2026-08-31_cnabu_new_sampling_true_full_single_process_v1`.
The archive has no immutable execution-date Git identity.

The adapted evaluator invokes `env.run(..., debug=False)` to keep the full
budget, sizes accumulators from MAX_ACTIONS, and explicitly converts CuPy
arrays before NumPy metric processing. Its metric calculations match the
September evaluations. This branch additionally removes the obsolete standalone
launcher and stops labelling every push as a collision in summary text: the
official pipeline uses action code 2 for every push. Collision statistics cannot
be recovered from those action codes.

## Run

Use the existing `mem_reproduction` Conda environment. Install this checkout
with its Python using `pip install --no-deps --no-build-isolation -e .` when
appropriate. This branch does not create an environment or bundle external
checkpoints, datasets, scene files, or calibration files. The official pipeline
expects its existing model/dataset assets under shelf_gym/scripts/model.
Keep a private copy of camera_matrices.npz because initialization can rewrite it.

An assets JSON is required, with `commit` equal to the official base above,
and `scenes` containing the 25 absolute scene paths. Preserve model/dataset hashes
and paths as additional JSON fields for provenance. Reuse the archived verified
asset manifests when reproducing the existing runs. The launcher records this
manifest path; keep the assets JSON with the run record.

Example (substitute your verified assets path and a new output directory):

```bash
CUDA_VISIBLE_DEVICES=0 EGL_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  /home/user/ehsanullahm1/miniconda3/envs/mem_reproduction/bin/python \
  tools/reproduction/run_mem.py --assets /path/to/mem_assets.json \
  --output /path/to/new_run
```

Default: no explicit seeding, lexically sorted scenes, 40 action slots.
Unset inherited PYTHONHASHSEED for the unseeded protocol. Add `--seed 23` for
controlled random initialization (set PYTHONHASHSEED=23 before starting Python
if matching the seeded experiment launcher). Seeded mode also configures cuDNN
and resets environment/push-sampler generators. Exact deterministic replay is
not guaranteed. Repeats should use separate processes and fresh output paths.
Use GPU UUIDs on shared Wonka and follow its GPU preflight/tmux conventions.
`--phase startup` or `--phase smoke` bounds a diagnostic to the first scene.
The runner rejects existing output directories and verifies imported shelf_gym
paths. No model training or checkpoint output is performed.

## Evidence and limits

Selected configuration, three seeded runs: semantic mIoU 0.710720 +/- 0.011487,
occupancy IoU 0.866724 +/- 0.002531 (sample SD). Legacy means: 0.709824 / 0.870847;
paper: 0.720 / 0.877. Selection and repeats are exploratory, not proof of exact
historical or statistical equivalence. Unseeded repeats have separate records.
See thesis_records/logs/2026-09-06_mem_ablation_completed.md and
2026-09-06_mem_unseeded_repeats.md. Bulk results stay outside this repository.

CPU checks:

```bash
/home/user/ehsanullahm1/miniconda3/envs/mem_reproduction/bin/python -m unittest discover -s tests -v
```
