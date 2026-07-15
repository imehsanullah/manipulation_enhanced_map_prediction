# Map Space Belief Prediction for<br/> Manipulation-Enhanced Mapping
![Overview figure](./images/overview.png)
This repository contains the code for reproducing the work of the RSS 2025 paper ["Map Space Belief Prediction for Manipulation-Enhanced Mapping"](https://arxiv.org/pdf/2502.20606), as well as the code for training the Calibrated Neural-Accelerated
Belief Update (CNABU) networks introduced in this paper. 

**This repository is a work in progress and currently contains the code required to run the proposed overall pipeline, as well as the data collection for network training.
The specific network training code and evaluation code used for the paper will be uploaded and updated in the near future.**

# Overview
![Overview figure](./images/architecture.png)
From a prior map belief, our pipeline predicts a map belief resulting from a set of candidate pushes. It then weighs the information gain from taking two consecutive independent views given the current belief (orange arrows) or taking a single observation given any of the predicted beliefs after pushing (blue arrows), selecting the path of highest cumulative information gain and taking its respective first action -- either taking the next best view or executing the best push. $IGV_t$ represents the best information gain obtainable from taking two distinct observation actions, whereas $IGM_t$ is the best information gain obtainable through a manipulation action followed by an observation action.


# Installation
In order to download the pre-trained CNABU models, please install ```git-lfs``` (apt install git-lfs).
To install and use the gym environment for this project we suggest using Anaconda3 or other virtual environments.
The code is tested with python >= 3.9 & <=3.12. 

- First, clone the repository including the submodule for scikit-geometry, as it is needed to run the code:
```bash
  git clone --recurse-submodules -j8 https://github.com/NilsDengler/manipulation_enhanced_map_prediction
  ```

- To install all the necessary packages, start your Conda environment run:
```bash
  conda activate YOUR_ENV_NAME
  cd YOUR_INSTALLATION_PATH/manipulation_enhanced_map_prediction
  git submodule update --init --recursive
  git submodule update --recursive --remote
  ./install.sh
  ```

# Structure
The project is structured as follows:
- ```shelf_gym``` contains the whole simulation structure.
  - ```shelf_gym/environments``` contains:
    - The base Pybullet environment ```base_environment.py```
    - The general generation of an ur5 robot in pybullet ```ur5_environment.py```
    - The world building script for the specific shelf environment. ```shelf_environment.py```
  - ```shelf_gym/meshes``` contains the meshes for the ur5, robotiq 85f2 gripper, YCB objects and environment specifics.
  - ```shelf_gym/scripts``` contains specific task related code which goes beyond the environment building.
    - ```shelf_gym/scripts/model``` contains the pre-tained CNABU-models, used fixed camera array, and a demo dataset used for utility functions
    - ```shelf_gym/scripts/data_generation``` contains the files to generate data for mapping and pushing to train the CNABUS
    - ```shelf_gym/scripts/model_training``` contains the files to train the CNABU models (**NOT updated yet**)
    - The **full pipeline script** ```run_cnabu_pipeline.py```, run this to replicate the papers results

  - ```shelf_gym/utils``` contains the utilities for the environment.

  
# Base Demo
To run a demo of the environment without executing the pipeline you can use the following command:
```bash 
  python shelf_gym/environments/shelf_environment.py
```

# Manipulation-Enhanced Mapping Demo
To run the demo of the full manipulation-enhanced mapping pipeline, as proposed in the paper, you can use the following command:
```bash 
  cd shelf_gym/scripts
  python run_cnabu_pipeline.py
```

# Data Collection Demo
To collect mapping or pushing data to train the CNABUS, run the following code:
```bash 
  cd shelf_gym/scripts/data_generation
  python map_collection.py #for map data only
  python push_collection.py #for map and push data pre- and post-psuh
```
Alternatively, you can collect the data for the [viewpoint push planning](https://github.com/NilsDengler/view-point-pushing) work by Dengler et al. using the following method:
```bash 
  cd shelf_gym/scripts/data_generation
  python dengler_iros_2023_map_collection.py 
```

# Thesis action-conditioned relation oracle

The thesis workspace adds a PyBullet adapter for generating physical
`blocks_access_to` evidence from the saved MEM shelf scenes. It replays the
saved object poses, evaluates a 3-by-3 frontal grasp/extraction grid, and
queries both robot and rigidly carried-target collisions for each action
stage. The portable scoring contract lives in the sibling `scene_graph_mem`
project.

Inspect one scene without writing anything:

```bash
PYTHONPATH=/home/user/ehsanullahm1/thesis/scene_graph_mem/src:. \
  /home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python \
  shelf_gym/scripts/inspect_action_conditioned_relation_oracle.py \
  --sample-dir /data/manipulation_map_data/raw/map_data/13/000000054/pre_action
```

Supplying `--output-dir` exports full continuous scores, validity masks,
per-trajectory blocker sets, the retained `geometry_pseudo_gt_v0` comparison,
and optional 3D diagnostics. Existing output directories are never
overwritten. The paired counterfactual pilot is run with
`shelf_gym/scripts/validate_action_conditioned_relation_counterfactuals.py`.

The v1 adapter separates shallow contact from penetration deeper than the
configured hard-contact threshold. Its randomized counterfactual check defines
clean extraction relative to each candidate's planned motion and rejects a
trial when the monitored blocker must be displaced significantly. It still
uses forced attachment after the grasp waypoint, so it validates access and
extraction rather than autonomous grasp closure.

For live candidate-level scene-graph inference, this project owns planner-side
action eligibility. `build_cnabu_runtime_candidate_action_mask` converts
ordered sparse learned-node support to robust, calibrated world boxes, solves
pregrasp/grasp/lift/extraction IK for the frozen 3-by-3 candidate family, and
checks the robot plus a temporary carried-target proxy against only known
fixed shelf/rack/table/wall geometry. The CNABU image-x reflection is handled
explicitly, and the default 5% support envelope avoids treating uncertain
one-voxel tails as exact collision geometry. The result is an aligned `[N,9]`
mask that uses no GT mask, simulator object id/pose, or dynamic-object query.
Dynamic source-object blockage remains the learned term in the sibling
`scene_graph_mem` project. The older IK-only adapter is retained for the
factorized-v1 ablation.

`shelf_gym/scripts/audit_cnabu_runtime_candidate_action_mask.py` checks the
frozen 100-record cause contract and can run a bounded live fidelity audit.
Its GT masks and oracle causes are loaded only after runtime inference for
matching and metrics; they never enter the planner mask.
`shelf_gym/scripts/export_cnabu_runtime_candidate_action_masks.py` is the
separate no-GT materializer for approved experiments. It writes a small mask
JSON per explicit record plus an enriched records JSON, refuses overwrites,
and supports `--validate-only` without writing outputs.

# Issue Tracker
 - 
