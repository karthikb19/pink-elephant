#!/bin/bash
# Play a fixed ladder of checkpoints against one Stockfish setting.
#
# Checkpoint-versus-checkpoint matches only ever say which of two nets is
# better. Stockfish at a pinned UCI Elo is an opponent that does not move
# between runs, so scores here are comparable across generations and across
# months, which is what makes a ladder out of what would otherwise be a pile of
# pairwise results.
#
#   ./scripts/run_stockfish_gauntlet.sh
#   ELO=2200 GAMES=40 ./scripts/run_stockfish_gauntlet.sh
#   DEVICE=mps ./scripts/run_stockfish_gauntlet.sh
#
# Checkpoints are pulled from the Modal training volume on first use and cached
# under data/modal-checkpoints/.
set -uo pipefail

ELO="${ELO:-2500}"
SIMULATIONS="${SIMULATIONS:-200}"
GAMES="${GAMES:-20}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-data/stockfish-gauntlet}"
CACHE="${CACHE:-data/modal-checkpoints}"
VOLUME="${VOLUME:-pink-elephant-training}"

# label | local filename | path on the Modal training volume
ENTRIES=(
"og-parent|og-parent.pt|runs/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10/checkpoints/20260810T041411Z-lichess-eval-v2-25m-from-10m-epoch10-epoch-000006-step-000110802.pt"
"gen2-5m-ep2|gen2-5m-ep2.pt|runs/20260823T032904Z-gen2-5m-anchor-030/checkpoints/20260823T032904Z-gen2-5m-anchor-030-epoch-000002-step-000010218.pt"
)

mkdir -p "$OUT_DIR" "$CACHE"
echo "Stockfish UCI Elo $ELO, $SIMULATIONS simulations, $GAMES games per checkpoint, device $DEVICE"
echo

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r label filename remote <<<"$entry"
  local_path="$CACHE/$filename"

  if [ ! -f "$local_path" ]; then
    echo "[$label] fetching $remote"
    if ! uv run modal volume get "$VOLUME" "$remote" "$local_path" >/dev/null 2>&1; then
      echo "[$label] FAILED to fetch; skipping"
      continue
    fi
  fi

  echo "[$label] playing $GAMES games against Stockfish $ELO"
  # --model-color alternate splits colours evenly, so a first-move advantage
  # cannot be mistaken for strength.
  uv run python scripts/play_stockfish.py \
    --checkpoint "$local_path" \
    --stockfish-elo "$ELO" \
    --model-simulations "$SIMULATIONS" \
    --games "$GAMES" \
    --model-color alternate \
    --device "$DEVICE" \
    2>&1 | tee "$OUT_DIR/${label}-elo${ELO}-sims${SIMULATIONS}.txt"
  echo
done

echo "transcripts under $OUT_DIR/"
