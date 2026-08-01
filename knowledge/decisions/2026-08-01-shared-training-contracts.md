# Shared training contracts

**Date:** 2026-08-01
**Status:** Accepted

## Context

PGN preprocessing and policy/value training should be developed in parallel.
The training loop must not depend directly on PGN parsing, `python-chess`, or
Parquet implementation details, while the data path must have a stable target
for its loader and collator.

The first supervised experiment needs legal-masked policy learning and a
signed scalar value target. Validation must compare policy learning with a
uniform-legal baseline and report both policy and value behavior.

## Decision

Define shared typed contracts in `pink_elephant.contracts`:

- `ExpertExample` represents one encoded pre-move position, legal actions,
  played action, game outcome, game ID, ply index, and split.
- `TrainingBatch` is the only data interface required by the training loop.
  It contains floating-point positions, a dense legal-action mask, played
  action indices, and signed value targets.
- `DatasetSchema` carries dataset, encoder, and action-schema versions.
- `JointLoss` groups differentiable total, policy, and value loss terms.
- `ValidationMetrics` defines the held-out metrics: legal-masked policy loss,
  uniform-legal policy loss, top-1/top-5 policy accuracy, value MSE, and value
  MAE.

The data stack will adapt PGN records into `ExpertExample` values and then into
processed shards. The loader/collator will adapt stored examples into
`TrainingBatch` values. The trainer will consume only `TrainingBatch` values.

## Alternatives

Keep the training loop coupled to a Parquet reader. This would make initial
development appear shorter but would prevent synthetic training tests and make
storage changes affect optimization code.

Pass sparse legal-action lists directly into the trainer. This saves a small
amount of batch memory but makes the trainer responsible for variable-length
batch handling. The dense legal mask is clearer for the first implementation
and matches the model's fixed policy output.

Use a categorical win/draw/loss value head. This may be useful later, but the
current model and data plan use one signed scalar value target, so changing it
now would expand the first experiment without a defined requirement.

## Consequences

PGN parsing and the training core can be implemented on separate branches from
the shared-contract commit. The training branch can use synthetic
`TrainingBatch` values before processed shards exist. The preprocessing branch
can validate its output independently before the integration branch connects
the loader to the trainer.

Runtime validation at the contract boundary catches malformed shapes, empty
legal-action sets, illegal played actions, invalid outcomes, and incompatible
metadata early. The dense legal mask is reconstructed by the collator; the
processed format can remain compact and sparse.

## Surface Areas

`src/pink_elephant/contracts.py`, future PGN/shard adapters, the dataset loader
and collator, joint-loss code, validation metrics, checkpoint metadata,
`tests/test_contracts.py`, and the expert-pretraining implementation scope.
