"""Evaluate pinned official pipelines using documented external metric harnesses."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Omit for unseeded evaluation.')
parser.add_argument('--arm', choices=['view'], default='view')
parser.add_argument('--kind', choices=['mem'], default='mem')
parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--assets', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--phase', choices=['startup', 'smoke', 'full'], default='full')
args = parser.parse_args()
sys.argv = [sys.argv[0]]  # Official FE-vMF constructor parses argv itself.
code = args.code.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.chdir(code)
sys.path.insert(0, str(code))
assets = json.loads(args.assets.read_text())
max_actions = 6 if args.phase == 'smoke' else 40
scenes = sorted(assets['scenes'])
if args.phase != 'full': scenes = scenes[:1]
started = time.time()

def write_json(name, data):
    (out / name).write_text(json.dumps(data, indent=2) + '\n')

manifest = dict(vars(args))
manifest = {k: str(v) if isinstance(v, Path) else v for k, v in manifest.items()}
manifest.update(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                max_actions=max_actions, scenes=scenes, seed=args.seed,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                egl_visible_devices=os.environ.get('EGL_VISIBLE_DEVICES'),
                python=sys.executable, checkpoint_writes=False, started_unix=started,
                policy='Corrected official MEM; semantic distance; ranking weight 2.0',
                corrections=['observation execution', 'post-push viewpoint', 'empty candidates', 'GEOS candidate guard', 'package discovery'],
                mem_run_debug=False, status='starting')
# The reproduction branch descends from the pinned official release.
assert assets['commit'] == '8ffa53f0c32e31f81ac9c1a75612b2e5a9443456'
subprocess.run(['git', 'merge-base', '--is-ancestor', assets['commit'], manifest['code_commit']], check=True)
write_json('manifest.json', manifest)
write_json('assets.json', assets)
(out / 'source.patch').write_text(subprocess.check_output(['git', 'diff', '--', '*.py'], text=True))
try:
    import numpy as np
    import cupy as cp
    import torch
    import shelf_gym
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    print('CUDA device:', torch.cuda.get_device_name(0), flush=True)
    print('Package:', shelf_gym.__file__, flush=True)
    evaluator_path = Path(__file__).parent / (args.kind + '_evaluator.py')
    spec = importlib.util.spec_from_file_location('official_evaluator', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    cls = evaluator.ManipulationEnhancedMapping
    imported = {name: module.__file__ for name, module in sys.modules.items()
                if name.startswith('shelf_gym') and getattr(module, '__file__', None)}
    assert all(Path(p).resolve().is_relative_to(code) for p in imported.values()), imported
    write_json('imported_sources.json', imported)

    def seed(env=None):
        if args.seed is None:
            return
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cp.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if env is not None:
            env.rng = np.random.default_rng(args.seed)
            env.ps.rng = np.random.default_rng(args.seed)

    seed()
    kwargs = {'use_uncertainty_informed_sampling': True} if args.kind == 'mem' else {'new_sampling': True}
    env = cls(render=False, save_dir=str(out / 'scratch'), **kwargs)
    env.action_budget = max_actions
    env.reset()
    seed(env)
    if args.phase == 'startup':
        env.restore_shelf_state(__import__('pickle').load(open(scenes[0], 'rb')))
        env.get_processed_array_and_gt_data()
        write_json('status.json', {'status': 'complete', 'phase': 'startup', 'wall_seconds': time.time()-started})
        print('STARTUP_OK', flush=True)
        env.close()
        sys.exit(0)

    # Disable only diagnostic PNG rendering; no predictions or action logic change.
    if args.kind == 'ms_mem':
        env.save_belief_map = lambda *a, **k: None
    original_store = env.store_results
    original_run = env.run
    context = {'scene_index': -1, 'scene': None, 'step': 0}
    trace = open(out / 'steps.jsonl', 'a', buffering=1)

    def store(*a, **kw):
        result = original_store(*a, **kw)
        data = result[0]
        context['step'] = len(data['pos_diffs'])
        event = dict(context, elapsed=time.time()-started,
                     pos_diff=float(data['pos_diffs'][-1]),
                     action_history_length=len(data['pushes']),
                     last_reported_action=int(data['pushes'][-1]) if data['pushes'] else None)
        trace.write(json.dumps(event)+'\n')
        write_json('status.json', dict(event, status='running'))
        print('PROGRESS', json.dumps(event), flush=True)
        return result

    def run(*a, **kw):
        context['scene_index'] += 1
        context['scene'] = str(kw.get('predefined_scene_dir', a[0] if a else None))
        context['step'] = 0
        write_json('status.json', dict(context, status='running'))
        result = original_run(*a, **kw)
        assert len(result['occupancy_map']) == max_actions
        write_json(f"scene_{context['scene_index']:03d}_actions.json", {
            'scene': context['scene'], 'actions': [int(v) for v in result['pushes']],
            'evaluated_steps': len(result['occupancy_map']),
            'note': 'Every push is labelled 2 by the official pipeline; do not infer collision counts.'})
        return result

    env.store_results = store
    env.run = run
    name = 'official_' + args.kind
    evaluator.EVAL_INTERVAL = 1
    evaluator.MAX_ACTIONS = max_actions
    evaluator.experiment_type = 'official'
    evaluator.name = name
    result_dir = out / 'results/official' / ('clean_start' if args.kind == 'mem' else 'grasping') / name
    result_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    df, summary, pushes, push_summary = evaluator.eval_and_process_policy(env, scenes, name)
    for label, frame in [('all', df), ('summary', summary), ('actions', pushes), ('action_summary', push_summary)]:
        frame.to_csv(out / (label + '.csv'), sep='|', index=False)
    write_json('status.json', {'status': 'complete', 'scenes_completed': len(scenes),
                               'wall_seconds': time.time()-started})
    print('EVALUATION_COMPLETE', flush=True)
    trace.close()
    env.close()
except Exception:
    write_json('status.json', {'status': 'failed', 'wall_seconds': time.time()-started,
                               'traceback': traceback.format_exc()})
    traceback.print_exc()
    sys.exit(1)
