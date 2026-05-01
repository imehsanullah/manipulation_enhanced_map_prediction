# Project Overview

This repository is a robotics research project for **Manipulation-Enhanced Mapping**.

In plain terms, it studies how a robot can understand a cluttered shelf by deciding whether it should:

- move its camera to a better viewpoint, or
- physically push objects to reveal hidden areas.

The project reproduces the paper **"Map Space Belief Prediction for Manipulation-Enhanced Mapping"** and implements the CNABU models: **Calibrated Neural-Accelerated Belief Updates**.

## Core Idea

The robot sees only part of a shelf because objects occlude each other. It maintains a probabilistic belief map of what it thinks is present in the scene, including object occupancy, semantic classes, and uncertainty.

At each step, the system compares two kinds of actions:

- **Observation action**: move to another camera viewpoint and gather more visual information.
- **Manipulation action**: push objects, then observe the scene after the push.

The planner estimates which action will reduce map uncertainty the most, then executes the first action of the best plan.

## What Happens In The Main Pipeline

The main pipeline is in:

```text
shelf_gym/scripts/run_cnabu_pipeline.py
```

The pipeline roughly does this:

1. Creates or restores a cluttered shelf scene in simulation.
2. Captures camera observations from a fixed camera array.
3. Builds an initial occupancy and semantic belief map.
4. Uses a map-completion neural model to predict hidden regions.
5. Samples possible robot push actions.
6. Uses a push-prediction neural model to estimate how the belief map would change after each push.
7. Computes expected information gain for camera viewpoints and push actions.
8. Chooses either a view change or a push.
9. Repeats until the map is sufficiently complete or the action budget is exhausted.

## Main Use

This project is used for research in:

- active perception,
- robotic manipulation for perception,
- semantic mapping under occlusion,
- next-best-view planning,
- push planning in cluttered shelves,
- uncertainty-aware map completion,
- simulation-based data generation for robot learning.

The practical motivation is a robot working in a cluttered household, grocery, or warehouse shelf. Instead of only looking passively, the robot can decide to move objects when that helps reveal hidden objects faster.

## Important Components

- `README.md`: high-level project description and run instructions.
- `shelf_gym/environments/shelf_environment.py`: PyBullet shelf environment with UR5 robot, Robotiq gripper, shelf, table, objects, and camera setup.
- `shelf_gym/scripts/run_cnabu_pipeline.py`: main manipulation-enhanced mapping planner.
- `shelf_gym/scripts/data_generation/map_collection.py`: generates map data from camera observations.
- `shelf_gym/scripts/data_generation/pushing_collection.py`: generates before/after push data.
- `shelf_gym/scripts/model_training/train_ycb_map_completion.py`: map-completion training code.
- `shelf_gym/scripts/model_training/train_ycb_push_prediction.py`: push-prediction training code.
- `shelf_gym/utils/models/UNet.py`: neural network architectures.
- `shelf_gym/utils/information_gain_utils.py`: information gain calculations for viewpoints.
- `shelf_gym/utils/mapping_utils.py`: heightmap, occupancy, semantic, and swept-volume mapping utilities.
- `visualization_attempts/`: scripts for visualizing belief maps, uncertainty, actions, and paper-style figures.

## Data And Models

The repository includes pretrained model artifacts under:

```text
shelf_gym/scripts/model/
```

Key files include:

- `model-5dburcae:v4.ckpt`: map-completion checkpoint.
- `push_predictor_new.ckpt`: push-prediction checkpoint.
- `camera_matrices.npz`: camera array calibration data.
- `dataset.hdf5`: demo/support dataset used by evaluation utilities.

## How To Run

Base environment demo:

```bash
python shelf_gym/environments/shelf_environment.py
```

Full manipulation-enhanced mapping pipeline:

```bash
cd shelf_gym/scripts
python run_cnabu_pipeline.py
```

Map data collection:

```bash
cd shelf_gym/scripts/data_generation
python map_collection.py
```

Push data collection:

```bash
cd shelf_gym/scripts/data_generation
python pushing_collection.py
```

## Important Notes

This is a research codebase, not a polished application. It has heavy robotics and machine-learning dependencies, including PyBullet, PyTorch/CUDA, CuPy, Open3D, Klampt, scikit-geometry, and CGAL.

The repository also appears to contain some duplicated nested project files. The active top-level Python package used by the inspected scripts is:

```text
shelf_gym
```

