from setuptools import setup, find_packages

setup(name='shelf_gym',
      version='0.0.3',
      packages=find_packages(include=['shelf_gym', 'shelf_gym.*']),
      install_requires=['gymnasium', 'pybullet', "tqdm",  "h5py", "attrdict", "scipy", "scikit-image", "imutils",
                        "opencv-python", "trimesh", "open3d", "tensorboard", "sb3-contrib", "stable_baselines3", "wandb",
                        "shapely==2.0.4", "voxelmap", "toppra", "seaborn","klampt","PyMCubes","pyfqmr", "cupy-cuda12x", "lightning", "PyOpenGL", "hdf5plugin", "torchvision", ]
)
 
