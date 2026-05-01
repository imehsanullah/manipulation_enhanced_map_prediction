## Instance Map Plots

These scripts visualize and analyze the ground truth instance maps stored in `gt_hms.npz` files.

### What are Instance Maps?

Instance maps are 2D top-down projections from multiple camera viewpoints that show **which specific object instance** is visible at each pixel location:

- **Pixel value = instance ID** (0, 1, 2, ...) for visible objects
- **Pixel value = -1** for occluded or empty regions
- **6 viewpoints** arranged in a grid pattern above the shelf

### Understanding Coverage

**Map Coverage** represents the percentage of the shelf surface visible from a viewpoint:

```
Coverage = (pixels with visible objects / total pixels) × 100
```

For example, with a 140×200 map (28,000 pixels):

- **Viewpoint 2**: 10,093 visible pixels = **36.0% coverage** (best)
- **Viewpoint 1**: 8,264 visible pixels = **29.5% coverage** (worst)

**Higher coverage** = camera sees more of the shelf surface (less occlusion)

### Available Scripts

#### 1. Basic Instance Maps with Rankings

```bash
python plots.py --mode instance_maps --save
```

**Output**: Grid of all 6 viewpoints with:

- Instance map visualization for each viewpoint
- Ranking badges (🥇, 🥈, 🥉) showing which viewpoints see most instances
- Coverage percentage for each viewpoint
- Summary bar chart comparing all viewpoints

**Best for**: Quick overview and comparison

---

#### 2. Detailed Viewpoint Analysis

```bash
python analyze_best_viewpoint.py --save
```

**Output**: Comprehensive text analysis including:

- Rankings by number of instances visible
- Coverage percentages
- Instance visibility patterns (always/sometimes/rarely visible)
- Identification of hard-to-see objects
- Summary plots

**Best for**: Understanding which viewpoint is optimal and why

**Example output**:

```
🏆 BEST VIEWPOINT: #2
   - Instances visible: 19
   - Pixels visible: 10,093
   - Coverage: 36.0%

📊 ALL VIEWPOINTS RANKED:
Rank   Viewpoint    Instances    Pixels       Coverage    
1      2            19           10,093       36.0%
2      3            19           9,334        33.3%
3      4            19           8,232        29.4%
...
```

---

#### 3. Coverage Calculation Explanation

```bash
python explain_coverage.py --save
```

**Output**: Step-by-step breakdown of coverage calculation:

- Shows the math for each viewpoint
- Annotated visualizations with calculation steps
- Detailed example for one viewpoint
- Explanation of what coverage means

**Best for**: Understanding exactly how coverage is computed

**Example calculation**:

```
Map dimensions: 140 × 200 = 28,000 total pixels
Pixels with instances: 8,264
Coverage = (8,264 / 28,000) × 100 = 29.5%
```

---

#### 4. Visible vs Occluded Visualization

```bash
python visualize_visible_mask.py --save
```

**Output**: Side-by-side comparison showing:

- **Left**: Instance map with colored object IDs
- **Right**: Visibility mask (green = visible, red = occluded)
- Exact pixel counts for visible/occluded regions

**Best for**: Visual understanding of what's visible vs hidden

---

### Common Command-Line Options

All scripts support:

```bash
--sample_id 000000005    # Analyze specific sample
--job_id 0               # Specify job ID
--save                   # Save plots to disk
--data_dir /path/to/data # Custom data directory
```

### Output Files

All visualizations are saved to:

```
visualization_attempts/plots/
├── instance_maps_000000000.png          # Basic rankings
├── viewpoint_analysis_000000000.png     # Detailed analysis
├── coverage_explanation_000000000.png   # Coverage math
└── visible_vs_occluded_000000000.png    # Visibility masks
```

### Key Metrics

When analyzing viewpoints, consider:

1. **Number of instances**: How many unique objects are visible?
2. **Coverage percentage**: How much of the shelf surface is visible?
3. **Pixel count**: Absolute number of visible pixels
4. **Consistency**: Which instances are visible from all/some viewpoints?

**Best viewpoint** = Highest instance count + highest coverage

---

