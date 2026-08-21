#!/usr/bin/env bash
# Play a candidate against its parent on one Modal GPU, reading both checkpoints
# from the training Volume so nothing is uploaded.
set -euo pipefail

cd "$(dirname "$0")/.."

PARENT_RUN=${PARENT_RUN:-20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10}
PARENT_CHECKPOINT=${PARENT_CHECKPOINT:-${PARENT_RUN}-epoch-000006-step-000110802.pt}

if [ -z "${CANDIDATE:-}" ]; then
  echo "Set CANDIDATE to a volume path, e.g. runs/<run-id>/checkpoints/<file>.pt" >&2
  exit 2
fi
PARENT=${PARENT:-runs/$PARENT_RUN/checkpoints/$PARENT_CHECKPOINT}

NAME_A=${NAME_A:-candidate}
NAME_B=${NAME_B:-parent}
POSITIONS=${POSITIONS:-256}
SIMULATIONS=${SIMULATIONS:-200}
SIMULATIONS_B=${SIMULATIONS_B:-0}
EXPLORATION=${EXPLORATION:-1.25}
MAX_PLIES=${MAX_PLIES:-300}
SEED=${SEED:-0}
OPENINGS=${OPENINGS:-data/openings/members_2025-10.jsonl}
OPENING_SEED=${OPENING_SEED:-0}
OUTPUT=${OUTPUT:-data/checkpoint-arena/modal-match.json}
# Give a side another checkpoint's value head while keeping its own policy.
VALUE_A=${VALUE_A:-}
VALUE_B=${VALUE_B:-}

# Openings are resolved on the client, so the book must exist locally.
test -f "$OPENINGS" || { echo "Missing opening book: $OPENINGS" >&2; exit 2; }

uv run modal run src/pink_elephant/checkpoint_match_modal.py \
  --checkpoint-a "$CANDIDATE" \
  --checkpoint-b "$PARENT" \
  --name-a "$NAME_A" \
  --name-b "$NAME_B" \
  --positions "$POSITIONS" \
  --simulations "$SIMULATIONS" \
  --simulations-b "$SIMULATIONS_B" \
  --exploration "$EXPLORATION" \
  --max-plies "$MAX_PLIES" \
  --seed "$SEED" \
  --openings "$OPENINGS" \
  --opening-seed "$OPENING_SEED" \
  --value-checkpoint-a "$VALUE_A" \
  --value-checkpoint-b "$VALUE_B" \
  --output "$OUTPUT"
