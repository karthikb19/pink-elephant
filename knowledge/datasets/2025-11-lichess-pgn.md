# Lichess PGN corpus: November 2025

## Purpose

This manifest identifies the local PGN input used for expert-game pretraining.
The PGNs and all generated data remain ignored under `data/`; this tracked file
lets another developer verify that they have the same source without adding
large artifacts to Git history.

## Local layout

```text
data/raw/lichess/2025-11/                 # complete corpus
data/fixtures/expert/2025-11-one-tenth.pgn # local pilot corpus
```

## Complete corpus

- Original source: `/Users/karthik/programming/dumbo/datasets/shards-2025-11`
- Local destination: `data/raw/lichess/2025-11/`
- Files: 29 PGN shards
- Games: 280,246
- Size: 270,640,461 bytes (258.1 MiB)

Verify the local source with:

```sh
find data/raw/lichess/2025-11 -maxdepth 1 -name '*.pgn' -type f -exec shasum -a 256 {} \; | sort
```

## Pilot corpus

The pilot is an ignored local PGN intended for initial parser and trainer
experiments. It is not a unit-test fixture.

- Selection: retain every tenth game in natural source-file order, using
  one-based global game indices divisible by ten
- Games: 28,024
- File: `data/fixtures/expert/2025-11-one-tenth.pgn`
- SHA-256: `ba6a12a2ce8d1403f23f53f7ad0004401840813fcc87f371e6afb61a063cc670`

Tiny, checked-in PGNs for parser unit tests belong separately under `tests/`
when ingestion is implemented.

## Source-file SHA-256

```text
1502a4e64e841e0a18ce2833f27009d7573d12e123b98789f9de08f83f94f669  shard_13.pgn
34e27d58b10d5329fd7c53d096897300732b830101ffc31f13e46cb719a78e6a  shard_16.pgn
5a83711b0b7dda50ab08c8b35c39910e8d1a1a2f115e5a0e6e9d1613ae9efad0  shard_3.pgn
5e4b3432274f57f1f3df9fa7448ff9ef36df06f37dc552d0cc15464e5dd1e6c5  shard_5.pgn
68b824c3dbc288543d82de5d7ac825c08767060d54b3209bcadb41b191ff2a40  shard_11.pgn
6ea5f51caeaf09008e82fa186546c282b132dc6073d1346fe20b1da08efe0755  shard_12.pgn
7b19eed0bb1f97c16705cf52268bf8981d58abe529f56af29379ebb987b97dfc  shard_24.pgn
867cb3f8f267ac40a1b17fa827048208a9dadec40fca6e0a6e6b26c9e5688f1c  shard_20.pgn
8b03a7ee4d215f7333253d8b63767eca39a13772b5ea54f99932fbc1751b7a07  shard_26.pgn
8da2cd30342fe0f3f326a8b227a3eb525814275c442f6ebd6ac2d19ab7b0bbc9  shard_4.pgn
904f0c77ee5b40e96b66d432820a5c6573866ee23da69920cdbcefdda45dd864  shard_2.pgn
910628b7d34fef2860a8f09c2ce48175ae5bf9a046b0d5a31da2038d0c22f4b8  shard_28.pgn
91cc4c8027e96b173ebbb6b52d9e125be11e4fbc05e13529f308486bf9dab8f6  shard_0.pgn
93ce6558b08a8a35492dc52efd944a9beac7c001c6b4faed6ed801d6fbd0482e  shard_1.pgn
97c750daa3da04815682f0f6316a4283e9ef3e2e0565900a56920df665beec42  shard_8.pgn
9f99a94ad3bd7ad1605cc368b3918a3654ded514b19f836a46cd1645193cba7b  shard_21.pgn
a97925fa791398686434500ec3f9708f40f87a4fac9ba1fcbcc6f7412f4dabd6  shard_10.pgn
b775f460d127b678e4e41ca0fae1fecc33961b32de40e7079c911f5f6c2c6727  shard_7.pgn
c262846f89286db9bb3a2abe152eca5f3a0b7dd39b2f5e180b079fbb6feca909  shard_25.pgn
c6dcf4428b9ac5251e6e08dd2090a3a565b4174e415361abb67bd30699331656  shard_27.pgn
cb96ef191f40009fa4753e431eda2007aa176e2d59abab81c60ad2bc440dd377  shard_19.pgn
cd59a4bafc9cc97703303cd032413e114cf163ad8b3790b2e7c1e0af02cf9b57  shard_6.pgn
da720f519a238d67816ff7bc9f6c4e7dc384bea05cd57cd87c9e83a30ff47ebe  shard_15.pgn
de3861e1f2aabd6ca8b0408e9afe422f693db70a37eee9fd666e996220605e4c  shard_14.pgn
e62ea1961f8722f23ec89555cb7b2032186835d566c698a561cbeb10aaeda0be  shard_22.pgn
e8f17f2426692e8097cdd22da1a8f405d6a1a0bbc0e3d26e4b2db2c66dfb1175  shard_18.pgn
ee7c19a992ea6fd0d45aea4dcef537c18de6a3340adf8ee012b300bdf317546d  shard_9.pgn
efdcdd398bc74855bb6c4df91ff6e313f21bceffbbffbca1b838f226eee64d2b  shard_17.pgn
f10df3c7d01e1907dc3d8c357ae1e25d37d9070d100a61f4be7bf2843f5d6ceb  shard_23.pgn
```
