# Belief Visualization - Quick Start Guide

**Location**: These visualization scripts are in the `visualization_attempts/` directory.

## 🚀 Fastest Way to Get Started

### Method 1: Use the Shell Script (Easiest!)

```bash
# From repo root
cd visualization_attempts

# Quick test (10 steps, ~2 minutes)
./run_visualization.sh --quick

# Full visualization (20 steps, ~10 minutes)
./run_visualization.sh --full

# With predefined scene
./run_visualization.sh --full --scene 5 --steps 25
```

### Method 2: Python Scripts

```bash
# From repo root
cd visualization_attempts

# Quick visualization
python quick_visualize.py --steps 10

# Full Figure 1 recreation
python visualize_belief_figure1.py --steps 20

# With specific scene
python visualize_belief_figure1.py --scene-id 5 --steps 25
```

---

## 📁 What Gets Created

All outputs are saved in `visualization_attempts/` subdirectories:

### Quick Mode Output:

```
visualization_attempts/quick_viz/
├── step_00.png      # Individual frames
├── step_01.png
├── ...
└── summary.png      # Combined view
```

### Full Mode Output:

```
visualization_attempts/belief_visualizations_TIMESTAMP/
├── figure1_style_comparison.png    # Main Figure 1 recreation
├── extended_comparison.png         # 5 timesteps view
├── frame_step_001.png             # Individual frames
├── frame_step_002.png
└── ...
```

---

## 🎨 What You'll See

Each visualization shows:

1. **Scene**: What the robot camera sees
2. **Robot's Belief**: What the robot thinks is there
  - Colored objects = high confidence predictions
  - Gray areas = uncertain regions
3. **Confidence Map**: How certain the robot is (red = confident)
4. **Uncertainty Map**: Where robot is unsure (yellow = uncertain)

---

## 📊 Understanding the Outputs

### Figure 1 Style (Main Output)

```
┌─────────────────────────────────────────────────┐
│  t=0          │  t=5          │  t=10           │
├───────────────┼───────────────┼─────────────────┤
│  Scene        │  Scene        │  Scene          │  ← Camera views
├───────────────┼───────────────┼─────────────────┤
│  Belief       │  Belief       │  Belief         │  ← What robot thinks
├───────────────┼───────────────┼─────────────────┤
│  [Action]     │  [Action]     │  [Action]       │  ← What robot did
└─────────────────────────────────────────────────┘
                Time →
```

- **Green boxes** = Observation actions (view changes)
- **Red boxes** = Manipulation actions (pushes)

---

## 🔧 Common Use Cases

### 1. Quick Test (Recommended First)

```bash
cd visualization_attempts
./run_visualization.sh --quick --steps 5
```

Output: `visualization_attempts/quick_viz/summary.png`

### 2. Paper Figure Recreation

```bash
cd visualization_attempts
./run_visualization.sh --full --steps 20
```

Output: `visualization_attempts/belief_visualizations_*/figure1_style_comparison.png`

### 3. Analyze Specific Scene

```bash
cd visualization_attempts
./run_visualization.sh --full --scene 10 --steps 30
```

### 4. Create Video

```bash
# Run full visualization first
cd visualization_attempts
./run_visualization.sh --full --steps 40

# Then create video
cd visualization_attempts/belief_visualizations_*/
ffmpeg -framerate 2 -pattern_type glob -i 'frame_*.png' \
       -c:v libx264 -pix_fmt yuv420p video.mp4
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```bash
cd visualization_attempts
./run_visualization.sh --quick --steps 5
```

### "No module named 'shelf_gym'"

```bash
# Make sure you're in the correct directory
cd /home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "Models not found"

Check that model files exist:

```bash
ls shelf_gym/scripts/model/
```

### Takes too long

- Use `--quick` mode instead of `--full`
- Reduce `--steps` to 5-10
- Quick mode: ~30 seconds per step
- Full mode: ~1 minute per step

---

## 📖 Files Reference

All files are located in `visualization_attempts/`:

| File                          | Purpose            | When to Use      |
| ----------------------------- | ------------------ | ---------------- |
| `run_visualization.sh`        | Convenience script | Always (easiest) |
| `quick_visualize.py`          | Fast testing       | Quick checks     |
| `visualize_belief_figure1.py` | Full pipeline      | Paper figures    |
| `belief_visualizer.py`        | Plotting library   | Custom code      |
| `VISUALIZATION_README.md`     | Full documentation | Detailed info    |


---

## 💡 Tips

1. **Start small**: Use `--quick --steps 5` first
2. **Save good scenes**: Note scene IDs that produce interesting results
3. **Compare**: Run same scene with/without pushing
4. **Videos**: Great for presentations - use 40+ steps
5. **Custom plots**: Import `belief_visualizer.py` in your own code

---

## 🎯 Next Steps

1. Run quick test:
  ```bash
   cd visualization_attempts
   ./run_visualization.sh --quick
  ```
2. Open result:
  ```bash
   xdg-open visualization_attempts/quick_viz/summary.png  # Linux
   # or
   open visualization_attempts/quick_viz/summary.png      # Mac
  ```
3. If satisfied, try full version:
  ```bash
   cd visualization_attempts
   ./run_visualization.sh --full --steps 20
  ```
4. Read full docs:
  ```bash
   cat visualization_attempts/VISUALIZATION_README.md
  ```

---

## 📚 Additional Resources

- **Full Documentation**: `VISUALIZATION_README.md`
- **Paper**: `2502.20606v3.pdf` (Figure 1 on page 1)
- **Code**: `shelf_gym/scripts/run_cnabu_pipeline.py` (lines 913-927 for visualization functions)

---

## ✅ Quick Checklist

Before running:

- In correct directory
- Models downloaded and in `shelf_gym/scripts/model/`
- CUDA available (`nvidia-smi` works)
- Enough disk space (~100MB per run)

After running:

- Check output directory created
- Open main PNG file
- Verify beliefs make sense
- Save interesting scene IDs

---

**Ready to start? Run this:**

```bash
./run_visualization.sh --quick
```

