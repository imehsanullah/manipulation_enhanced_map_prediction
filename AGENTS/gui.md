# GUI Notes

The project can show a PyBullet GUI when the environment is created with:

```python
render=True
```

The display session was verified with:

```bash
printenv DISPLAY
```

The active display was:

```text
:1
```

## Working GUI Demo

A bounded PyBullet GUI demo was run successfully with:

```bash
conda run -n manipulation_map python -c "import time; from shelf_gym.environments.shelf_environment import ShelfEnv; env=ShelfEnv(render=True, show_vis=False, use_ycb=True); print('GUI_DEMO_STARTED'); start=time.time();
while time.time()-start < 45:
    env.step_simulation(env.per_step_iterations)
    time.sleep(1/15)
print('GUI_DEMO_DONE'); env.close()"
```

This opened a PyBullet window on the user's screen, initialized the shelf and UR5 robot scene, ran briefly, and closed cleanly.

Successful output included:

```text
I will render
Environment Initialized
GUI_DEMO_STARTED
GUI_DEMO_DONE
```

## Important Distinction

The warning:

```text
PyQt5 is not available... try running "pip3 install PyQt5"
Neither QT nor GLUT are available... visualization disabled
```

comes from Klampt's optional visualizer, not from PyBullet itself.

PyBullet GUI still worked through X11/GLX with the NVIDIA GPU.

The successful GUI output included:

```text
X11 functions dynamically loaded using dlopen/dlsym OK!
Creating context
Created GL 3.3 context
Direct GLX rendering context obtained
GL_VENDOR=NVIDIA Corporation
GL_RENDERER=NVIDIA GeForce RTX 4080 SUPER/PCIe/SSE2
```

## Headless vs GUI

For headless/offscreen execution:

```python
render=False
```

This uses EGL rendering and does not open a window.

For visible PyBullet GUI execution:

```python
render=True
```

This uses `p.GUI` in `shelf_gym/environments/base_environment.py`.

## Caution

The direct base environment script:

```bash
conda run -n manipulation_map python shelf_gym/environments/shelf_environment.py
```

opens a GUI but runs an infinite loop:

```python
while True:
    env.step_simulation(env.per_step_iterations)
```

Prefer a bounded command for demos unless an interactive long-running PyBullet session is intended.

