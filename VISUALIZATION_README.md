# Belief Visualization Scripts

This directory contains scripts to visualize the robot's belief evolution during the Manipulation-Enhanced Mapping (MEM) pipeline, recreating visualizations similar to Figure 1 from the paper.

## Quick Start

### Option 1: Simple Quick Visualization (Recommended for First Try)

Run a quick 10-step visualization with minimal setup:

```bash
python quick_visualize.py
```

With custom number of steps:
```bash
python quick_visualize.py --steps 20
```

**Output**: Creates `./quick_viz/` directory with:
- Individual frames for each step
- Summary figure showing progression

---

### Option 2: Full Figure 1 Style Visualization

This creates publication-quality visualizations similar to Figure 1 in the paper.

#### Basic usage (random scene):
```bash
python visualize_belief_figure1.py --steps 20
```

#### With predefined scene:
```bash
python visualize_belief_figure1.py --scene-id 5 --steps 25
```

#### With custom scene file:
```bash
python visualize_belief_figure1.py --scene /path/to/scene_data.p --steps 20
```

**Output**: Creates `./belief_visualizations_TIMESTAMP/` directory with:
- `figure1_style_comparison.png` - Main Figure 1 style visualization
- `extended_comparison.png` - Extended view with 5 timesteps
- `frame_step_XXX.png` - Individual frames for each step

---

## Output Examples

### Figure 1 Style Comparison
Shows progression across 3 key timesteps (t, t+1, t+2):
- Top row: Scene images
- Middle row: Robot's belief (semantic map)
- Bottom row: Action descriptions
- Highlights when pushes occur vs observations

### Individual Frames
Each frame shows:
- **Scene**: Camera view from robot
- **Robot's Belief**: Semantic map with colored objects
- **Confidence Map**: How certain the robot is (red = high confidence)
- **Uncertainty Map**: Where the robot is uncertain (yellow = high uncertainty)

---

## Understanding the Visualizations

### Color Coding in Semantic Belief:
- **Bright colored objects**: High confidence predictions
- **Gray/faded areas**: Low confidence or uncertain regions
- **White areas**: Unmapped/unknown space

### Action Types:
- **Green box**: Observation action (viewpoint change)
- **Red box**: Manipulation action (push)

### What the Robot "Believes":
The visualizations show what the robot **thinks** is in the scene based on:
1. Direct observations from camera
2. Predictions from the learned CNABU models
3. Reasoning about occluded areas

---

## Creating Videos

After running the full visualization script, create a video:

```bash
# Navigate to output directory
cd belief_visualizations_TIMESTAMP/

# Create video (requires ffmpeg)
ffmpeg -framerate 2 -pattern_type glob -i 'frame_*.png' \
       -c:v libx264 -pix_fmt yuv420p belief_evolution.mp4
```

Adjust `-framerate` to control video speed (higher = faster).

---

## Script Details

### `quick_visualize.py`
- **Purpose**: Fast, simple visualization for testing
- **Execution time**: ~2-5 minutes for 10 steps
- **Use case**: Quick check of belief evolution
- **Actions**: Observation only (no pushes)

### `visualize_belief_figure1.py`
- **Purpose**: Publication-quality Figure 1 recreation
- **Execution time**: ~5-15 minutes for 20 steps
- **Use case**: Paper figures, detailed analysis
- **Actions**: Both observations and pushes (full MEM pipeline)
- **Features**:
  - Scene + belief side-by-side
  - Multiple comparison views
  - Action annotations
  - Confidence/uncertainty maps

---

## Command Line Arguments

### `visualize_belief_figure1.py`

```bash
python visualize_belief_figure1.py [OPTIONS]

Options:
  --scene PATH          Path to predefined scene file (.p)
  --scene-dir PATH      Directory containing scene files
                        (default: ./data/Hard_scenes/scenes/)
  --scene-id INT        Scene ID to load (e.g., 5 loads scene_data_5.p)
  --steps INT           Maximum number of steps to run (default: 20)
```

### `quick_visualize.py`

```bash
python quick_visualize.py [OPTIONS]

Options:
  --steps INT           Number of steps to run (default: 10)
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce the number of sampled pushes:
```python
# Edit visualize_belief_figure1.py, line ~180
num_points=40  # Reduce from 80 to 40
```

### Issue: No scene image captured
**Solution**: Ensure `render=True` is set in ManipulationEnhancedMapping initialization.

### Issue: Black/empty belief visualizations
**Solution**:
1. Check that models are properly loaded
2. Verify the model checkpoint paths in `run_cnabu_pipeline.py` lines 38-41

### Issue: Script takes too long
**Solution**:
- Use `quick_visualize.py` instead
- Reduce `--steps` parameter
- Use smaller scenes

---

## File Structure

```
manipulation_enhanced_map_prediction/
├── visualize_belief_figure1.py          # Main visualization script
├── quick_visualize.py                    # Quick test script
├── VISUALIZATION_README.md               # This file
├── shelf_gym/scripts/
│   └── run_cnabu_pipeline.py            # Core MEM pipeline
└── belief_visualizations_TIMESTAMP/      # Output directory (created when run)
    ├── figure1_style_comparison.png
    ├── extended_comparison.png
    └── frame_step_XXX.png
```

---

## Integration with Paper Figure 1

The visualizations recreate Figure 1 from the paper:

**Paper Figure 1 shows**:
- Time step progression (t → t+1 → t+2)
- Scene images at each timestep
- Robot's belief (top-down semantic map)
- Actions taken (View Change, Move and Push)
- Annotation of what changed

**Our visualization provides**:
- ✅ Same layout and progression
- ✅ Scene + belief side-by-side
- ✅ Action type labels
- ✅ Timestep markers
- ✅ Additional confidence/uncertainty maps
- ✅ Extended 5-timestep comparison

---

## Customization

### Modify Visualization Style

Edit `visualize_belief_figure1.py`:

```python
# Change figure size
fig = plt.figure(figsize=(20, 12))  # Line ~200

# Change color maps
ax.imshow(sem_conf, cmap='viridis')  # Line ~250

# Save more/fewer snapshots
if step % 2 == 0:  # Save every 2nd step instead of all
    visualizer.save_individual_frame(...)
```

### Add Custom Metrics

Add to `capture_snapshot()` method:
```python
# Calculate custom metric
metric_value = calculate_my_metric(previous_semantic_map)
snapshot['my_metric'] = metric_value
```

---

## Requirements

- PyTorch with CUDA
- matplotlib
- numpy
- All dependencies from main project

---

## Citation

If you use these visualizations in your work, please cite:

```bibtex
@article{marques2025map,
  title={Map Space Belief Prediction for Manipulation-Enhanced Mapping},
  author={Marques, Joao Marcos Correia and Dengler, Nils and others},
  journal={arXiv preprint arXiv:2502.20606},
  year={2025}
}
```

---

## Additional Resources

- **Paper**: [arXiv:2502.20606](https://arxiv.org/abs/2502.20606)
- **GitHub**: Check main README for repository link
- **Video examples**: See supplementary material

---

## Questions?

For issues or questions:
1. Check the troubleshooting section above
2. Review the main project README
3. Open an issue on the GitHub repository
