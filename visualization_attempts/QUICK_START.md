# Quick Start Guide - Map Data Visualization

All plotting scripts are now working and have been fixed to use the non-interactive 'Agg' backend to avoid GUI issues.

**Location**: These visualization scripts are in the `visualization_attempts/` directory.

## Generated Plots

All generated plots are saved in the `visualization_attempts/plots/` directory. Here's what you can create:

### Basic Examples (`example_plotting.py`)
- `camera_array_overview.png` - Overview of 6 camera views with height maps, semantic maps, and depth images
- `ground_truth_overview.png` - Complete ground truth visualization including 2D/3D maps and cross-sections
- `comparison.png` - Side-by-side comparison of predicted vs ground truth maps
- `all_cameras_grid.png` - Grid showing all 50 camera height maps

### Advanced Analysis (`advanced_plotting_examples.py`)
- `height_distribution.png` - Histogram of height values across all cameras
- `semantic_distribution.png` - Bar chart showing pixel count per semantic class
- `multi_camera_comparison.png` - 2x2 comparison of different camera viewpoints
- `occupancy_vs_height.png` - Plot showing occupancy at different height levels
- `depth_per_camera.png` - Line plot of mean depth across all 300 cameras
- `cross_sections.png` - 3D voxel cross-sections in X-Z, Y-Z, and X-Y planes

## Usage

### 1. Quick Data Summary
```bash
python3 data_summary.py
```
Shows all available samples and data statistics.

### 2. Basic Plotting Examples
```bash
python3 example_plotting.py
```
Generates 4 basic visualization plots.

### 3. Advanced Analysis
```bash
python3 advanced_plotting_examples.py
```
Generates 6 advanced analysis plots with statistics.

### 4. Command-Line Tool
```bash
# Overview of camera array
python3 plots.py --mode overview --save

# Ground truth visualization
python3 plots.py --mode ground_truth --sample_id 000000003 --save

# Comparison plot
python3 plots.py --mode comparison --camera_idx 0 --save

# All cameras grid
python3 plots.py --mode all_cameras --save
```

### 5. Custom Python Scripts

```python
import matplotlib
matplotlib.use('Agg')  # Always set this first!
import os
import sys
sys.path.insert(0, 'visualization_attempts')
from plots import MapDataVisualizer

# Path is automatically resolved relative to repo root
viz = MapDataVisualizer()
samples = viz.find_available_samples(job_id=0)

# Load data
hms_data = viz.load_hms_data(samples[0])
gt_data = viz.load_gt_data(samples[0])

# Create plots (always provide save_path)
viz.plot_camera_array_overview(hms_data, save_path='visualization_attempts/plots/my_plot.png')
viz.plot_ground_truth_overview(gt_data, save_path='visualization_attempts/plots/my_gt_plot.png')
```

## Important Notes

1. **Always use `matplotlib.use('Agg')`** before importing pyplot to avoid GUI issues
2. **All plots are saved to files** - there is no interactive display
3. **The `plots/` directory** is automatically created if it doesn't exist
4. **Large datasets**: Use `max_cameras` parameter to limit the number of cameras plotted
5. **Run from repo root**: Execute scripts from the main repository root or use absolute paths

## Data Structure

Your dataset contains:
- **12 samples** in job 0
- **300 camera views** per sample
- **Height map resolution**: 140x200 pixels
- **Depth image resolution**: 480x640 pixels
- **3D voxel map**: 140x200x102 (X×Y×Z)
- **12 unique semantic classes**

## File Sizes

Each sample contains:
- `hms.npz`: ~16 MB (all camera data)
- `gt_hms.npz`: ~0.13 MB (ground truth)
- `camera_matrices.npz`: ~0.03 MB (camera parameters)
- `placed_objects.pkl`: ~0.01 MB (object placement info)

## Troubleshooting

**If you get a segmentation fault:**
- Make sure you have `matplotlib.use('Agg')` at the top of your script
- Don't call `plt.show()` when using the Agg backend

**If plots are not generated:**
- Check that the `plots/` directory exists
- Ensure you're providing `save_path` parameter to plotting functions

**For interactive plotting:**
- Set environment variable: `export MPLBACKEND=TkAgg`
- Use `show=True` parameter in plotting functions
- This only works if you have a display available
