# Map Data Visualization Guide

This guide explains how to use `plots.py` to visualize the .npz files in `shelf_gym/data/map_data`.

## Quick Start

### Command-line Usage

The `plots.py` script can be used directly from the command line:

```bash
# View overview of camera array (first 6 cameras)
python plots.py --mode overview

# View ground truth data
python plots.py --mode ground_truth

# Compare prediction vs ground truth
python plots.py --mode comparison --camera_idx 0

# View all cameras in a grid
python plots.py --mode all_cameras

# Specify a specific sample
python plots.py --mode overview --sample_id 000000008

# Save plots to disk
python plots.py --mode overview --save
```

### Python API Usage

You can also use the `MapDataVisualizer` class in your own scripts:

```python
from plots import MapDataVisualizer

# Initialize visualizer
viz = MapDataVisualizer(data_dir="shelf_gym/data/map_data")

# Find available samples
samples = viz.find_available_samples(job_id=0)
sample_dir = samples[0]

# Load data
hms_data = viz.load_hms_data(sample_dir)
gt_data = viz.load_gt_data(sample_dir)

# Create visualizations
viz.plot_camera_array_overview(hms_data)
viz.plot_ground_truth_overview(gt_data)
viz.plot_comparison(sample_dir, camera_idx=0)
viz.plot_all_cameras_grid(hms_data)
```

## Data Structure

### Height Map Data (hms.npz)

- **hms**: (N, 140, 200, 2) - Height maps from N cameras
  - Channel 0: Height values in meters
  - Channel 1: Border mask
- **dilated_hms**: (N, 140, 200, 2) - Morphologically dilated height maps
- **semantic_hms**: (N, 140, 200) - Semantic labels for each pixel
- **semantics**: (N, 480, 640, 2) - Full resolution semantic images
- **depths**: (N, 480, 640) - Depth images from cameras

### Ground Truth Data (gt_hms.npz)

- **gt_hms**: (2, 140, 200) - Ground truth 2D maps
  - Channel 0: Occupancy probability
  - Channel 1: Height map
- **hm3d**: (140, 200, 102) - 3D voxel occupancy map
- **semantic_2d**: (140, 200) - 2D semantic class map
- **semantic_3d**: (140, 200, 102) - 3D semantic voxel map

## Visualization Modes

### 1. Camera Array Overview
Shows height maps, semantic maps, and depth images from multiple cameras side-by-side.

```bash
python plots.py --mode overview
```

### 2. Ground Truth Overview
Displays all ground truth data including 2D maps, 3D voxel maps, and cross-sections.

```bash
python plots.py --mode ground_truth
```

### 3. Prediction vs Ground Truth Comparison
Compares predicted height maps from cameras against ground truth.

```bash
python plots.py --mode comparison --camera_idx 0
```

### 4. All Cameras Grid
Shows all camera views in a compact grid layout.

```bash
python plots.py --mode all_cameras
```

## Example Script

Run the example script to see all visualization modes:

```bash
python example_plotting.py
```

## Available Methods

### MapDataVisualizer Class

- `find_available_samples(job_id)`: List all available sample directories
- `load_hms_data(sample_dir)`: Load height map data from hms.npz
- `load_gt_data(sample_dir)`: Load ground truth data from gt_hms.npz
- `plot_single_heightmap(hm, camera_idx, ax, title)`: Plot a single height map
- `plot_semantic_map(semantic_map, ax, title)`: Plot semantic map with colors
- `plot_depth_image(depth, camera_idx, ax)`: Plot depth image
- `plot_camera_array_overview(hms_data, camera_indices, save_path)`: Multi-camera overview
- `plot_ground_truth_overview(gt_data, save_path)`: Ground truth visualization
- `plot_comparison(sample_dir, camera_idx, save_path)`: Prediction vs GT comparison
- `plot_all_cameras_grid(hms_data, max_cameras, save_path)`: Grid of all cameras

## Tips

1. **Large datasets**: Use `max_cameras` parameter to limit the number of cameras plotted
2. **Custom camera views**: Specify `camera_indices` list to plot specific cameras
3. **Saving plots**: Use `--save` flag or `save_path` parameter to save figures
4. **Interactive exploration**: Use the Python API for interactive exploration in Jupyter notebooks

## Requirements

The script uses the following packages (already present in your environment):
- numpy
- matplotlib
- seaborn
- scipy (for mode calculation in 3D semantic maps)
