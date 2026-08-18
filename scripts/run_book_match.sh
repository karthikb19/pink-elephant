#!/usr/bin/env bash
# Play the self-play candidate against its parent from popular human opening positions.
set -euo pipefail

cd "$(dirname "$0")/.."

CANDIDATE=${CANDIDATE:-'https://modal.com/storage/karthikb19/main/volumes/pink-elephant-training/runs/20260818T205016Z-self-play-iteration-official-1/checkpoints/20260818T205016Z-self-play-iteration-official-1-epoch-000005-step-000005685.pt'}
PARENT=${PARENT:-'https://modal.com/storage/karthikb19/main/volumes/pink-elephant-training/runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt'}
NAME_A=${NAME_A:-self-play-epoch-5}
NAME_B=${NAME_B:-parent-epoch-6}

BOOK_SOURCE_URL=${BOOK_SOURCE_URL:-https://raw.githubusercontent.com/jesung/engine-equal-human-unequal/refs/heads/main/data/members_2025-10.jsonl}
BOOK_SOURCE=${BOOK_SOURCE:-data/openings/members_2025-10.jsonl}
BOOK=${BOOK:-data/openings/human-2025-10-30.jsonl}
POSITIONS=${POSITIONS:-30}
BOOK_SEED=${BOOK_SEED:-0}
MIN_OPENING_COUNT=${MIN_OPENING_COUNT:-500}
MIN_OPENING_PLY=${MIN_OPENING_PLY:-4}
MAX_OPENING_PLY=${MAX_OPENING_PLY:-12}

SIMULATIONS=${SIMULATIONS:-128}
EXPLORATION=${EXPLORATION:-1.25}
OPENING_TEMPERATURE=${OPENING_TEMPERATURE:-1.0}
TEMPERATURE_CUTOFF_PLY=${TEMPERATURE_CUTOFF_PLY:-0}
SEED=${SEED:-0}
MAX_PLIES=${MAX_PLIES:-256}
DEVICE=${DEVICE:-cpu}
TORCH_THREADS=${TORCH_THREADS:-4}

# Each position is played twice so both models get both colors from the same opening.
GAMES=$((POSITIONS * 2))

if [ ! -f "$BOOK_SOURCE" ]; then
  echo "Downloading opening source: $BOOK_SOURCE_URL"
  mkdir -p "$(dirname "$BOOK_SOURCE")"
  curl -fsSL -o "$BOOK_SOURCE" "$BOOK_SOURCE_URL"
fi

uv run pe-openings "$BOOK_SOURCE" "$BOOK" \
  --count "$POSITIONS" \
  --seed "$BOOK_SEED" \
  --min-count "$MIN_OPENING_COUNT" \
  --min-ply "$MIN_OPENING_PLY" \
  --max-ply "$MAX_OPENING_PLY"

echo
echo "Playing $GAMES games ($POSITIONS openings x 2 colors) at $SIMULATIONS simulations"
echo

uv run play-checkpoints \
  "$CANDIDATE" \
  "$PARENT" \
  --name-a "$NAME_A" \
  --name-b "$NAME_B" \
  --games "$GAMES" \
  --openings "$BOOK" \
  --simulations "$SIMULATIONS" \
  --exploration "$EXPLORATION" \
  --opening-temperature "$OPENING_TEMPERATURE" \
  --temperature-cutoff-ply "$TEMPERATURE_CUTOFF_PLY" \
  --seed "$SEED" \
  --max-plies "$MAX_PLIES" \
  --device "$DEVICE" \
  --torch-threads "$TORCH_THREADS"
