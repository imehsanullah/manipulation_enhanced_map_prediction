# Training Explanation

Training in this project is split into **two related stages**:

1. map-completion training,
2. push-prediction training.

The first model learns to complete a hidden map from partial observations. The second model learns to predict how that map changes after a robot push.

## 1. Map-Completion Training

The map-completion model answers:

> Given a few camera observations of a cluttered shelf, what is probably hidden behind the visible objects?

Main file:

```text
shelf_gym/scripts/model_training/train_ycb_map_completion.py
```

The data comes from simulated shelf scenes. For each scene, the dataset stores:

- camera height maps,
- semantic maps,
- depth maps,
- free-space voxels,
- occupied-space voxels,
- ground-truth 3D occupancy,
- ground-truth semantic labels.

The dataset loader is in:

```text
shelf_gym/utils/learning_utils/datasets.py
```

The model receives several partial observations. Each observation tells the model:

- these voxels are free,
- these voxels are occupied,
- this semantic class was observed at this map location.

The model updates a belief map step by step.

The output is:

- a **3D occupancy belief**: occupied vs free for each voxel,
- a **2D semantic belief**: object class prediction per map cell.

## Evidential Prediction

The map-completion model is not trained as a simple classifier. It is an **evidential model**.

That means it predicts parameters of probability distributions instead of only predicting a single class probability.

In this project:

- occupancy is represented with Beta-like evidence,
- semantics are represented with Dirichlet-like evidence.

This is important because the planner needs uncertainty estimates. It needs to know not only *what* the model predicts, but also *how confident* the model is.

The relevant loss functions are in:

```text
shelf_gym/utils/learning_utils/losses.py
```

The map-completion loss combines:

- occupancy prediction error,
- semantic classification error,
- uncertainty/evidence regularization using KL divergence.

So the model is penalized for being wrong and also for being overconfident when it should be uncertain.

## 2. Push-Prediction Training

The push-prediction model answers:

> If the robot performs this push, what will the shelf belief look like afterward?

Main file:

```text
shelf_gym/scripts/model_training/train_ycb_push_prediction.py
```

This model depends on the map-completion model. It first uses the trained map-completion model to create a current belief map, then adds push-specific information.

The push dataset contains:

- pre-push observations,
- pre-push ground truth,
- post-push ground truth,
- swept volume of the gripper/object motion,
- push start/end parametrization,
- difference map showing which cells changed.

The push model input is built in:

```text
shelf_gym/utils/learning_utils/data_preprocessing.py
```

The key class is:

```text
PushDataPrepper
```

It does this:

1. Runs the trained map-completion model on the pre-push observations.
2. Gets the current occupancy and semantic belief.
3. Adds the swept-volume map.
4. Adds push endpoint channels.
5. Sends the combined representation into the push model.

The push model architecture is in:

```text
shelf_gym/utils/models/UNet.py
```

The relevant model class is:

```text
PushSemanticUNet
```

The push model outputs:

- predicted post-push occupancy map,
- predicted post-push semantic map,
- predicted difference/change map.

Its training loss combines:

- semantic loss after the push,
- occupancy loss after the push,
- difference-map loss,
- consistency loss for areas that should not change.

The consistency loss matters because most of the shelf should remain unchanged after a push. The model should predict changes only where the push actually affects the scene.

## How Training Data Is Generated

Training data is generated in simulation.

Map data generation:

```text
shelf_gym/scripts/data_generation/map_collection.py
```

This script creates random cluttered shelf scenes, captures camera views, builds height maps and semantic maps, and stores simulator ground truth.

Push data generation:

```text
shelf_gym/scripts/data_generation/pushing_collection.py
```

This script does:

1. Generate a cluttered shelf.
2. Save pre-push map, camera, and ground-truth data.
3. Execute a push.
4. Save post-push ground truth.
5. Save swept volume and push parameters.

Because everything happens in simulation, labels can be generated automatically from the simulator state.

## Practical Training Commands

Map-completion training:

```bash
cd shelf_gym/scripts/model_training
python train_ycb_map_completion.py
```

Push-prediction training:

```bash
cd shelf_gym/scripts/model_training
python train_ycb_push_prediction.py
```

## Required Training Datasets

The training scripts expect HDF5 datasets such as:

```text
map_completion_fine_tune.hdf5
unbiased_push_dataset.hdf5
```

In the current checkout, these expected training datasets were not found during inspection.

The repository does include pretrained checkpoints under:

```text
shelf_gym/scripts/model/
```

Important pretrained files:

- `model-5dburcae:v4.ckpt`: map-completion checkpoint.
- `push_predictor_new.ckpt`: push-prediction checkpoint.
- `camera_matrices.npz`: camera calibration data.
- `dataset.hdf5`: demo/support dataset.

So the normal demo runs with pretrained models rather than requiring training from scratch.

## Summary

Training teaches one neural network to complete a hidden semantic/occupancy map from partial camera views. Then it teaches a second neural network to predict how that belief map changes after a robot push.

The planner later uses these trained models to decide whether the robot should look from another viewpoint or push objects to reveal hidden parts of the shelf.

