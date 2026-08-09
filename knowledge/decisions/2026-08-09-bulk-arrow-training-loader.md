# Bulk Arrow training loader

**Date:** 2026-08-09
**Status:** Accepted

## Context

Training spent roughly 44% of steady batch time waiting for the CPU loader. The processed-data
reader converted every Arrow value through scalar Python objects and constructed a validated
`ExpertExample` before recreating the same data as batch tensors.

## Decision

Read Parquet record batches into typed NumPy arrays in bulk, validate row contracts with array
operations, and let `ExpertBatchLoader` gather tensors directly from those arrays. Preserve the
public example iterator as a compatibility adapter over the bulk representation and preserve the
existing bounded-buffer shuffle algorithm and random-number sequence.

## Alternatives

Keeping scalar conversion retained the measured bottleneck. Loading complete shards as dense
tensors would simplify batching but use substantially more memory, especially for legal-action
masks. Adding multiprocessing before reducing Python work would add operational complexity without
addressing the underlying conversion cost.

## Consequences

Training avoids row-level Arrow scalar conversion and per-example contract objects while retaining
streaming, deterministic shuffle, schema validation, and final partial batches. Arrow-backed arrays
remain alive while shuffle-buffer references need them, so CPU memory use can exceed the logical
row payload of the shuffle buffer by several reader batches.

## Surface Areas

Processed Parquet reading in `shards.py`, training batch construction in `dataset.py`, and dataset
equivalence and malformed-data tests.
