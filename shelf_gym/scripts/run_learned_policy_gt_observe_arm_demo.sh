#!/usr/bin/env bash
set -euo pipefail

# Re-runs the learned CNABU/MEM policy-loop demo with:
# - PyBullet GUI
# - live learned_component_splitter scene graph window
# - simulator GT top-down diagnostic panel
# - UR5 dummy_camera_link motion before each observation
#
# Optional overrides:
#   ACTION_BUDGET=10 HOLD_SEC=60 ./shelf_gym/scripts/run_learned_policy_gt_observe_arm_demo.sh
#   SHOW_BELIEF_PANEL=1 ./shelf_gym/scripts/run_learned_policy_gt_observe_arm_demo.sh
#   DIAGNOSTICS_DIR=/tmp/my_demo ./shelf_gym/scripts/run_learned_policy_gt_observe_arm_demo.sh
#   HEADLESS=1 ./shelf_gym/scripts/run_learned_policy_gt_observe_arm_demo.sh

REPO_ROOT="${REPO_ROOT:-/home/user/ehsanullahm1/thesis/manipulation_enhanced_map_prediction}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/ehsanullahm1/miniconda3/envs/manipulation_map/bin/python}"
DIAGNOSTICS_PARENT="${DIAGNOSTICS_PARENT:-/home/user/ehsanullahm1/thesis/thesis_records/diagnostics}"

ACTION_BUDGET="${ACTION_BUDGET:-6}"
PUSH_NUM_POINTS="${PUSH_NUM_POINTS:-20}"
PUSH_IG_SKIP="${PUSH_IG_SKIP:-5}"
OBSERVE_ARM_PAUSE_SEC="${OBSERVE_ARM_PAUSE_SEC:-1.2}"
SLEEP_SEC="${SLEEP_SEC:-2.0}"
HOLD_SEC="${HOLD_SEC:-45}"
DEVICE="${DEVICE:-cpu}"
HEADLESS="${HEADLESS:-0}"
SHOW_BELIEF_PANEL="${SHOW_BELIEF_PANEL:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "MEM repo not found: $REPO_ROOT" >&2
  exit 1
fi

if [[ "$HEADLESS" != "1" && -z "${DISPLAY:-}" ]]; then
  export DISPLAY=":1"
  echo "DISPLAY was unset; defaulting to DISPLAY=:1"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
diagnostics_dir="${DIAGNOSTICS_DIR:-${DIAGNOSTICS_PARENT}/mem_cnabu_policy_learned_gt_observe_arm_gui_${timestamp}}"
mkdir -p "$diagnostics_dir"

cd "$REPO_ROOT"

cmd=(
  "$PYTHON_BIN" -u
  "$REPO_ROOT/shelf_gym/scripts/run_cnabu_scene_graph_live_demo.py"
  --scene-graph-mode learned_component_splitter
  --device "$DEVICE"
  --policy-loop
  --action-budget "$ACTION_BUDGET"
  --show-gt-panel
  --move-arm-for-observe
  --observe-arm-pause-sec "$OBSERVE_ARM_PAUSE_SEC"
  --push-num-points "$PUSH_NUM_POINTS"
  --push-ig-skip "$PUSH_IG_SKIP"
  --save-diagnostics
  --diagnostics-dir "$diagnostics_dir"
  --sleep-sec "$SLEEP_SEC"
  --hold-sec "$HOLD_SEC"
)

if [[ "$HEADLESS" != "1" ]]; then
  cmd+=(--render --show-graph)
fi

if [[ "$SHOW_BELIEF_PANEL" == "1" ]]; then
  cmd+=(--show-belief-panel)
fi

if [[ -n "${CHECKPOINT:-}" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($EXTRA_ARGS)
  cmd+=("${extra_args[@]}")
fi

echo "Running learned CNABU/MEM policy-loop demo"
echo "Repo: $REPO_ROOT"
echo "Diagnostics: $diagnostics_dir"
echo "Command:"
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
