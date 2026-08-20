# Checkpoint match sources

## Context

Checkpoint comparisons need to work from both existing local files and artifacts in Modal Volumes without redownloading large models on every invocation.

## Decision

The checkpoint match CLI accepts local paths, `modal://VOLUME/REMOTE_PATH` references, and Modal storage URLs. Remote files are downloaded atomically into an identity-hashed local cache, and match results record source strings, local paths, hashes, parameters, scores, and PGNs.

## Alternatives

Require users to download checkpoints manually, or access Modal directly through the SDK inside the match process.

## Consequences

Matches are repeatable and cached while Modal authentication remains delegated to the installed CLI. An explicit remote reference is required when a local path does not exist.

## Surface Areas

`scripts/play_checkpoints.py`, `pink_elephant.checkpoint_match_cli`, arena game orchestration, local checkpoint cache, and checkpoint match output artifacts.
