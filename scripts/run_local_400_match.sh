#!/usr/bin/env bash
# Play the anchored candidate against the parent locally, both at 400 simulations.
set -euo pipefail

cd "$(dirname "$0")/.."

VOLUME=${VOLUME:-pink-elephant-training}

# Volume-relative path to the anchored candidate's checkpoint. No default: pass it in.
#   CANDIDATE_REMOTE='runs/<anchored-run-id>/checkpoints/<file>.pt' ./scripts/run_local_400_match.sh
CANDIDATE_REMOTE=${CANDIDATE_REMOTE:?set CANDIDATE_REMOTE to the anchored checkpoint path on the $VOLUME volume}
PARENT_REMOTE=${PARENT_REMOTE:-'runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt'}

CANDIDATE_LOCAL="checkpoints/$(basename "$CANDIDATE_REMOTE")"
PARENT_LOCAL="checkpoints/$(basename "$PARENT_REMOTE")"

for pair in "$CANDIDATE_REMOTE|$CANDIDATE_LOCAL" "$PARENT_REMOTE|$PARENT_LOCAL"; do
  remote=${pair%%|*}
  local_path=${pair##*|}
  if [ ! -f "$local_path" ]; then
    echo "Downloading $remote"
    uv run modal volume get "$VOLUME" "$remote" "$local_path"
  fi
done

POSITIONS=${POSITIONS:-5}   # openings; games are twice this
SIMULATIONS=${SIMULATIONS:-400}
DEVICE=${DEVICE:-cpu}
PGN_OUT=${PGN_OUT:-data/checkpoint-arena/local-400sim-$(date +%Y%m%dT%H%M%S).pgn}
mkdir -p "$(dirname "$PGN_OUT")"

uv run scripts/play_checkpoint_match.py \
  "$CANDIDATE_LOCAL" \
  "$PARENT_LOCAL" \
  --name-a "${NAME_A:-anchor030}" \
  --name-b "${NAME_B:-parent}" \
  --positions "$POSITIONS" \
  --simulations "$SIMULATIONS" \
  --device "$DEVICE" \
  --pgn-out "$PGN_OUT"
