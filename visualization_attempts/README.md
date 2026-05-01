# Visualization Attempts

This directory contains all visualization scripts and their outputs for the Manipulation-Enhanced Map Prediction project.

## Structure

```
visualization_attempts/
├── README.md                           # This file
├── PLOTTING_README.md                  # Original plotting documentation
├── QUICK_START.md                      # Quick start for map data visualization
├── VISUALIZATION_QUICKSTART.md         # Quick start for belief visualization
├── VISUALIZATION_README.md             # Full belief visualization documentation
│
├── plots.py                            # Core map data visualization library
├── example_plotting.py                 # Basic map data examples
├── advanced_plotting_examples.py       # Advanced map data analysis
├── data_summary.py                     # Data statistics utility
│
├── belief_visualizer.py                # Belief plotting library
├── quick_visualize.py                  # Quick belief visualization
├── visualize_belief_figure1.py         # Full Figure 1 style visualization
├── run_visualization.sh                # Convenience script for belief viz
│
├── extract_instances.py                # Instance segmentation from beliefs
├── advanced_instance_extraction.py     # Advanced instance extraction methods
│
├── plots/                              # Output: Map data visualizations
├── quick_viz/                          # Output: Quick belief visualizations
├── belief_visualizations_*/            # Output: Full belief visualizations
├── instance_extraction/                # Output: Instance segmentation
└── advanced_instance_comparison/       # Output: Method comparisons
```

## Two Main Visualization Systems

### System 1: Map Data Visualization

Visualizes pre-existing map data from `.npz` files (height maps, semantic maps, depth images, ground truth).

**Scripts**:

- `plots.py` - Core visualization library
- `example_plotting.py` - Basic examples
- `advanced_plotting_examples.py` - Advanced analysis
- `data_summary.py` - Data statistics

**Usage**:

```bash
cd visualization_attempts
python3 example_plotting.py
python3 advanced_plotting_examples.py
python3 data_summary.py
```

**Output**: `visualization_attempts/plots/` directory

---

### System 2: Belief Visualization

Visualizes the robot's belief evolution during the MEM (Manipulation-Enhanced Mapping) pipeline.

**Scripts**:

- `belief_visualizer.py` - Standalone plotting library
- `quick_visualize.py` - Fast testing (10 steps, ~2-5 min)
- `visualize_belief_figure1.py` - Full Figure 1 recreation (20 steps, ~5-15 min)
- `run_visualization.sh` - Convenience wrapper
- `extract_instances.py` - Instance segmentation
- `advanced_instance_extraction.py` - Advanced instance methods (6 methods)

**Usage**:

```bash
cd visualization_attempts

# Quick test
./run_visualization.sh --quick --steps 10

# Full visualization
./run_visualization.sh --full --steps 20

# With specific scene
./run_visualization.sh --full --scene 5 --steps 25
```

**Output**: 

- `visualization_attempts/quick_viz/` (quick mode)
- `visualization_attempts/belief_visualizations_*/` (full mode)
- `visualization_attempts/instance_extraction/`
- `visualization_attempts/advanced_instance_comparison/`

---

## Running from Repo Root

All scripts are designed to be run from the repository root:

```bash
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction
cd visualization_attempts

# Now run any script
python3 example_plotting.py
./run_visualization.sh --quick
```

The scripts automatically resolve paths relative to the repository root.

## Requirements

- PyTorch with CUDA
- matplotlib
- numpy
- scipy
- scikit-image
- opencv-python (for advanced instance extraction)
- All dependencies from main project

## Troubleshooting

### "No module named 'shelf_gym'"

```bash
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd visualization_attempts
python3 example_plotting.py
```

### "FileNotFoundError: shelf_gym/data/map_data"

Make sure you're running from the repo root or have the data directory in the expected location.

### "CUDA out of memory"

Use fewer steps or smaller scenes:

```bash
./run_visualization.sh --quick --steps 5
```

## Documentation

- **QUICK_START.md** - Map data visualization quick start
- **VISUALIZATION_QUICKSTART.md** - Belief visualization quick start
- **VISUALIZATION_README.md** - Full belief visualization documentation
- **PLOTTING_README.md** - Original plotting documentation

