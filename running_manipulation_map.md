# Running Manipulation-Enhanced Mapping Demo

This guide contains all the steps needed to successfully run the manipulation-enhanced mapping demo with Python 3.12.

## Prerequisites

- Conda environment (e.g., `manipulation_map`)
- Python 3.12
- Git with submodules initialized

## Installation Steps

### 1. Fix setup.py for Package Discovery

Edit the `setup.py` file to explicitly specify packages:

```bash
# The setup.py should be modified to include:
from setuptools import setup, find_packages

setup(name='shelf_gym',
      version='0.0.3',
      packages=find_packages(include=['shelf_gym', 'shelf_gym.*']),
      install_requires=[...])
```

Or use this sed command to make the change:
```bash
sed -i 's/from setuptools import setup/from setuptools import setup, find_packages/' setup.py
sed -i 's/setup(name/setup(name/' setup.py
```

### 2. Install shelf_gym Package

```bash
cd /path/to/manipulation_enhanced_map_prediction
pip install -e .
```

### 3. Install CGAL Library (Version 5.6.1)

CGAL 6.0+ has compatibility issues with scikit-geometry, so we need version 5.6.1:

```bash
conda install -y -c conda-forge "cgal<6.0"
```

This will install:
- cgal-5.6.1
- cgal-cpp-5.6.1
- eigen-3.4.0
- gmp-6.3.0
- mpfr-4.2.1
- And other required dependencies

### 4. Install scikit-geometry from Submodule

Navigate to the scikit-geometry submodule and install it:

```bash
cd shelf_gym/utils/scikit-geometry

# Modify setup.py to allow newer pybind11 versions
sed -i 's/pybind11>=2.3,<2.8/pybind11>=2.3/g' setup.py

# Build and install
python -m pip wheel . -w wheel
python -m pip install -e .
```

**Note:** This compilation step takes several minutes (5-10 minutes) as it compiles C++ code with CGAL. This is a one-time process.

### 5. Navigate Back to Project Root

```bash
cd /path/to/manipulation_enhanced_map_prediction
```

## Running the Demo

### Base Demo

Run the base environment demo:

```bash
python shelf_gym/environments/shelf_environment.py
```

**Expected Output:**
- PyBullet simulation engine will initialize
- UR5 robot with Robotiq 85 gripper will load (23 links, 23 joints)
- Shelf environment with collision geometries will be created
- OpenGL rendering context will be created

**Note:** If running on a headless server or without display, you'll see X11/GLUT errors at the end. This is expected and doesn't affect the core functionality. The environment initializes successfully.

### Full Pipeline Demo

To run the complete manipulation-enhanced mapping pipeline:

```bash
cd shelf_gym/scripts
python run_cnabu_pipeline.py
```

### Data Collection

To collect training data:

```bash
cd shelf_gym/scripts/data_generation

# For map data only
python map_collection.py

# For map and push data (pre- and post-push)
python push_collection.py
```

## Troubleshooting

### Issue: ModuleNotFoundError for shelf_gym
**Solution:** Make sure you ran `pip install -e .` from the project root.

### Issue: ModuleNotFoundError for skgeom
**Solution:** Install scikit-geometry following step 4 above.

### Issue: CGAL compilation errors
**Solution:**
- Ensure CGAL 5.6.1 is installed (not 6.0+)
- Check that conda environment has the correct CGAL version: `conda list cgal`
- If needed, downgrade: `conda install -y -c conda-forge "cgal<6.0"`

### Issue: scikit-geometry compilation takes too long
**Solution:** This is expected. The C++ compilation with CGAL takes 5-10 minutes. Be patient, it's a one-time compilation.

### Issue: X11 connection errors or GLUT errors
**Solution:** This is expected when running without a display. The core simulation still works. To suppress these:
- Run with DISPLAY= prefix: `DISPLAY= python shelf_environment.py`
- Or use virtual display: `xvfb-run python shelf_environment.py`

## Verification

To verify everything is installed correctly:

```python
# Test imports
python -c "import shelf_gym; print('shelf_gym OK')"
python -c "import skgeom; print('skgeom OK')"
python -c "import pybullet; print('pybullet OK')"
```

All should print "OK" without errors.

## Environment Details

- **Conda Environment:** manipulation_map
- **Python Version:** 3.12
- **CGAL Version:** 5.6.1
- **Main Dependencies:** gymnasium, pybullet, CGAL, scikit-geometry, open3d, pytorch

## Summary of Key Changes Made

1. Modified `setup.py` to use `find_packages()` for proper package discovery
2. Installed CGAL 5.6.1 (not 6.0) for compatibility with scikit-geometry
3. Modified scikit-geometry's setup.py to allow pybind11>=2.3 (removed upper bound)
4. Compiled scikit-geometry with CGAL support

---

Last Updated: 2025-11-13

