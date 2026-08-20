# Generate self-play data with more search than the match uses

**Date:** 2026-08-20
**Status:** Accepted

## Context

Self-play fine-tuning was making the model worse. The first run scored 0.300
against its supervised parent over 30 games and the second 0.045 over 11; both
confidence intervals sat entirely below parity. Two separate causes turned up.

The value head was being destroyed. It had been pretrained on `tanh`-scaled
Stockfish evaluations and was now regressing terminal outcomes in `{-1, 0, 1}`,
a target that is constant across a whole game. A frozen anchor of 20,000
held-out engine evaluations showed correlation with those evaluations falling
0.878 -> 0.797 -> 0.703 across epochs while predicted spread rose to 1.107
against the parent's 0.903. The head was saturating toward the labels' rails,
not collapsing to their mean, which is what hard outcome targets predict.

The games themselves were also poor. About 10% of moves lost at least two pawns
of material, and six times more search did not change that. The cause was move
selection rather than search: play was sampled proportional to root visits, so a
move holding one visit out of 200 was played 0.5% of the time, and a position
offers dozens of them. One observed game played `15...Nd6` at 0.5% while search
wanted `Be7` at 32%, losing a piece. The network already knew the move was bad.

## Decision

Three changes, each measured independently.

Blend the value target: `q_ratio * root_value + (1 - q_ratio) * outcome` at 0.5,
following Leela's `q_ratio`, with `value_weight` reduced from 1.0 to 0.25.

Never play a move below `min_visit_fraction` of the most visited move's visits,
defaulting to 0.10. The threshold is relative so it adapts to how decisive the
position is. The policy target keeps the full distribution; only the played move
is constrained, so no training signal is lost.

Buy opening variety from the start position rather than from playing moves the
search dislikes: 50% of games start from the human opening book, 20% from the
standard position, 30% from archived engine-evaluated positions banded by
evaluation magnitude.

Add forced playouts and policy target pruning (KataGo 3.2), so Dirichlet noise
improves exploration without teaching the policy to predict the extra playouts.

## Alternatives

Lowering `dirichlet_fraction` below AlphaZero's 0.25 was measured and rejected.
Noise off versus on, same checkpoint and seed, moved the blunder rate 8.04% to
12.76%, but the code applies noise at every move with alpha 0.3 exactly as the
paper specifies for chess, and exploration is meant to cost move quality.

Raising the temperature to widen openings was rejected for the same reason: it
buys variety by playing worse moves. The start book buys it for free.

An absolute visit threshold, as in KataGo's `chosenMovePrune = 1`, would not
have blocked the observed blunder, which had exactly one visit. KataGo pairs
that with `chosenMoveTemperature = 0.10`, which crushes the tail before pruning
matters; this repository samples at temperature 1.0, so the threshold has to be
relative.

## Consequences

Corpus quality improved: blunders 10.13% -> 6.33%, severe blunders 3.84% ->
2.46%, mean game length 73 -> 102 plies, draws 13% -> 28%. White's score from
balanced book positions came out at 0.506 over 6,151 games, where 0.500 is fair.

The value head stopped degrading. Anchor correlation held at 0.854-0.862 across
all five epochs against the parent's 0.878, with spread at 0.906-0.933 against
0.903. Sign agreement matched or beat the parent.

Strength reached parity but no further. Over 512 games, epoch 1 scored 0.4844
with a 95% interval of [0.452, 0.517] and epoch 4 scored 0.4746 with [0.441,
0.508]. A paired comparison over the same openings put the difference between
them at +0.0098, interval [-0.035, +0.055], so epoch choice does not matter and
the policy cross-entropy that degrades every epoch does not cost games.

The reason is that a model cannot exceed its own teacher. Data was generated at
200 simulations and matches were played at 200, so training pulled the raw
policy toward 200-simulation quality while the parent reached the same place
through search. Measured directly: the same checkpoint at 800 simulations beat
itself at 200 by 164W 35D 1L over 200 games, a score of 0.9075 or about +397
Elo, interval [0.880, 0.935].

Generation should therefore use a much larger simulation budget than the match
or deployment does, which is the only way the target carries information the
student cannot already reach. At equal budgets the loop has no gradient to
climb, which is exactly the parity that was measured twice.

Two things this does not establish. Whether the gain persists at higher budgets
is unknown, since search returns usually diminish and 800 -> 3200 may be worth
much less than 200 -> 800. And a 400 Elo stronger teacher will not transfer 400
Elo to the student, because distillation is lossy.

Supervised mixing remains unbuilt. Policy cross-entropy against held-out
supervised rows degrades every epoch, which is forgetting rather than
overfitting, and mixing Stockfish-labelled rows into training batches targets it
directly while the value head keeps learning outcomes.

## Surface Areas

`mcts.py` and `rust/pe-search` for forced playouts, policy target pruning, and
the visit floor, kept conformant by differential tests. `self_play/generation`
for the start-position mix and the recorded root value.
`self_play/learning/replay.py` for the blended target.
`checkpoint_match_modal.py` and `match_host.py` for batched matches, including
per-model simulation budgets. `scripts/value_anchor.py` for drift measurement
and `scripts/inspect_self_play_games.py` for corpus quality.
