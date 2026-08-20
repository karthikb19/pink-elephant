# Self-play sample games

Each game records the `seed` and `game_id` it was generated with, so a game can be located in the corpus or re-run locally without copying the movetext. A local replay reproduces the same position and search settings, not the same move sequence, because the native engine and the Python reference draw from different random streams.

3,508 games available, 3,349 within 25-250 plies.

| start | games | mean plies |
| --- | --- | --- |
| startpos | 725 | 114.1 |
| book | 1,698 | 114.2 |
| archive | 926 | 81.7 |
| other | 0 | 0.0 |

## startpos

**0-1** — checkmate, 160 plies

`seed 8734485349463169380` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000305`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 d6 6. g4 h6 7. h4 Nc6 8. Be3 e5 9. Nf5 Be6 10.
Nd5 Nxd5 11. exd5 Qa5+ 12. c3 Qxd5 13. Qxd5 Bxd5 14. Rh2 g6 15. Ng3 O-O-O 16. h5 Be6 17. f3 d5
18. O-O-O d4 19. Bd2 Bxa2 20. Ne4 g5 21. Be1 Bb3 22. Rd3 Bc4 23. Rhd2 Bxd3 24. Bxd3 dxc3 25.
bxc3 Be7 26. Ng3 Kc7 27. Nf5 Bc5 28. Kc2 Ne7 29. Ng3 Be3 30. Re2 Bf4 31. Bc4 f6 32. Ne4 f5 33.
Nf2 Nc8 34. gxf5 Nd6 35. Be6 Rhf8 36. Ng4 Nxf5 37. Nxe5 Ne3+ 38. Kb3 Bxe5 39. Rxe3 Bf4 40. Re4
Rd1 41. Bg4 Rfd8 42. Kb2 Rc1 43. Bg3 Bxg3 44. Kxc1 b5 45. Re6 Rd6 46. Bf5 Rxe6 47. Bxe6 Kd6 48.
Bc8 Be1 49. Kc2 a5 50. Ba6 Kc6 51. Bc8 a4 52. Bf5 Kd6 53. Kb2 Ke5 54. Bd7 Kf4 55. Bxb5 Kxf3 56.
Bxa4 g4 57. Bc6+ Kf2 58. c4 g3 59. Kb3 Ba5 60. Ka4 Bc7 61. c5 g2 62. Bxg2 Kxg2 63. Kb5 Kf3 64.
Kc6 Bf4 65. Kd7 Kg4 66. c6 Kxh5 67. Ke7 Kg4 68. Kd7 h5 69. Kc8 h4 70. c7 Bxc7 71. Kxc7 h3 72.
Kb6 h2 73. Kc7 h1=Q 74. Kd6 Kf5 75. Kc5 Ke5 76. Kc4 Qf3 77. Kb4 Kd5 78. Kb5 Qb3+ 79. Ka5 Kc6
80. Ka6 Qb6#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 8734485349463169380 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**0-1** — checkmate, 130 plies

`seed 6993775961256028663` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000022`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. d4 d5 2. Nf3 Nf6 3. c4 e6 4. Nc3 Nbd7 5. e3 Be7 6. b3 dxc4 7. bxc4 O-O 8. Be2 b6 9. O-O Bb7
10. Bb2 Re8 11. Qc2 Bd6 12. e4 e5 13. d5 Qe7 14. Nb5 Nc5 15. Nd2 a5 16. Ba3 Bc8 17. Nxd6 cxd6
18. Rab1 Qc7 19. f3 Bd7 20. Nb3 Nh5 21. Nxc5 bxc5 22. g3 Nf6 23. Bb2 a4 24. Bc3 Reb8 25. Rxb8+
Rxb8 26. Rb1 h5 27. Rxb8+ Qxb8 28. Qc1 Qb6 29. h4 Bh3 30. Kh2 Bd7 31. Kg1 Qb7 32. Ba5 a3 33.
Bc3 Bc8 34. Kf2 Qa7 35. Bd3 Kf8 36. Bd2 Ne8 37. Qc3 Bd7 38. Qb3 Qa4 39. Qb1 Qa8 40. Qb6 Qa4 41.
Ke2 Qa8 42. Ba5 Ke7 43. Kd2 Ba4 44. Bc3 Bd7 45. f4 f6 46. Ba5 Qc8 47. Be2 Bg4 48. Bd3 Qa8 49.
fxe5 fxe5 50. Bc2 Nf6 51. Bc3 Nd7 52. Qc7 Qf8 53. Ba4 Qf2+ 54. Kc1 Qe3+ 55. Bd2 Qg1+ 56. Kc2
Bd1+ 57. Kb1 Bxa4+ 58. Bc1 Qd4 59. Bg5+ Ke8 60. Qd8+ Kf7 61. Qe7+ Kg8 62. Qe8+ Nf8 63. Qxe5
Qxe5 64. Kc1 Qc3+ 65. Kb1 Qb2#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 6993775961256028663 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**0-1** — checkmate, 98 plies

`seed 16701181779106177823` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000685`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 d6 5. c4 Nf6 6. Nc3 g6 7. Be2 Bg7 8. Be3 O-O 9. O-O Be6
10. Nxe6 fxe6 11. h4 Nh5 12. Bxh5 gxh5 13. Qxh5 Qa5 14. Qe2 Rae8 15. Rad1 Kh8 16. f4 Bxc3 17.
bxc3 Qxc3 18. h5 Rg8 19. Rf3 Qf6 20. c5 dxc5 21. Bxc5 Rg4 22. Qf2 Reg8 23. Rd7 e5 24. fxe5 Qg5
25. Rd2 Qxe5 26. h6 Rxe4 27. Be3 Reg4 28. Rf8 Rxf8 29. Qxf8+ Rg8 30. Qf2 Rg6 31. Rd5 Qxd5 32.
Qf8+ Rg8 33. Qf2 Qd1+ 34. Kh2 Qd6+ 35. Kg1 Qd1+ 36. Kh2 Rd8 37. Qf7 Qd6+ 38. Kg1 Qg3 39. Bd4+
Rxd4 40. Qf8+ Qg8 41. Qf5 Rd1+ 42. Kh2 Rd6 43. Kg1 Rxh6 44. a4 Rf6 45. Qe4 Qg3 46. Qe2 Nd4 47.
Qe4 Qf2+ 48. Kh2 Rh6+ 49. Qh4 Rxh4#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 16701181779106177823 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1-0** — checkmate, 141 plies

`seed 7187474742065797911` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000025`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. d4 d5 2. c4 Nf6 3. cxd5 c6 4. dxc6 Nxc6 5. Nf3 Bg4 6. Nc3 Bxf3 7. exf3 e6 8. Qa4 Qxd4 9. Bb5
Qxa4 10. Bxa4 a6 11. Bxc6+ bxc6 12. O-O a5 13. Be3 Kd7 14. Na4 Nd5 15. b3 Bd6 16. Rfd1 Kc7 17.
Rac1 Ba3 18. Rc4 Nxe3 19. fxe3 Rhd8 20. Rcd4 Rxd4 21. Rxd4 e5 22. Rd3 f5 23. e4 fxe4 24. fxe4
Rd8 25. Rxd8 Kxd8 26. Kf2 Ke7 27. Ke2 Bc1 28. Nb6 Bf4 29. g3 Bc1 30. Nc4 Kd7 31. Nxa5 Kd6 32.
Nc4+ Kc5 33. Nxe5 Kd4 34. Nxc6+ Kxe4 35. b4 Kd5 36. b5 Ba3 37. Kd3 Kc5 38. Nd4 Bb2 39. Nf3 Kxb5
40. Ng5 h6 41. Ne6 Kc6 42. a4 Kb6 43. Kc4 Ba1 44. Nd4 Bb2 45. Nb3 Bf6 46. Kd5 Bc3 47. g4 Kc7
48. h4 Be1 49. h5 Kd7 50. a5 Kc7 51. Ke6 Bc3 52. Kf5 Kb7 53. Nc5+ Kc6 54. a6 Bb2 55. a7 Kxc5
56. a8=Q Bc3 57. Qf8+ Kc4 58. Qf7+ Kd3 59. Qb3 Kd4 60. Kg6 Bd2 61. Kxg7 Bc1 62. Qd1+ Ke3 63.
Qxc1+ Kd4 64. Qxh6 Kc3 65. Qg5 Kd3 66. h6 Ke4 67. h7 Kd3 68. h8=Q Kc2 69. Qe3 Kb1 70. Qh2 Ka1
71. Qe1#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 7187474742065797911 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**0-1** — checkmate, 182 plies

`seed 17349441883479601868` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000722`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. d4 d5 2. Nf3 e6 3. e3 Nf6 4. Bd3 b6 5. c4 Be7 6. Nc3 O-O 7. cxd5 exd5 8. O-O Re8 9. Ne5 c5
10. b3 Bf8 11. Bb2 Nbd7 12. Rc1 Nxe5 13. dxe5 Rxe5 14. Ne2 Re6 15. Nf4 Rd6 16. Be5 g6 17. h3
Bb7 18. Re1 Qe7 19. Bb2 Rad8 20. Qc2 Bg7 21. Rcd1 a5 22. a4 Ne4 23. Bxg7 Kxg7 24. Ne2 Qe5 25.
f4 Qe7 26. Bxe4 dxe4 27. Rxd6 Rxd6 28. Qb2+ Kg8 29. Nc1 Qd8 30. Qe5 Rd2 31. Ne2 Ba6 32. Ng3 f5
33. Rc1 Bd3 34. Qc3 Ra2 35. Qe5 Kf7 36. Rd1 Rc2 37. Kh2 Rf2 38. Kh1 Qf6 39. Qc7+ Kg8 40. Kh2 h5
41. Kg1 Rc2 42. Qb8+ Kh7 43. Qc7+ Kh6 44. Qb8 c4 45. bxc4 h4 46. Nh1 Re2 47. Nf2 Rxe3 48. Qe8
Re2 49. Qd7 Rc2 50. Nxd3 exd3 51. Qxd3 Rc3 52. Qe2 Ra3 53. Qe5 Qxe5 54. fxe5 Rxa4 55. e6 Rxc4
56. e7 Re4 57. Rd7 a4 58. Kf2 Kg5 59. Rb7 Kf4 60. Rxb6 Rxe7 61. Ra6 Re4 62. Rxg6 Rc4 63. Ra6
Rd4 64. Ra7 Rb4 65. Ra8 Rb2+ 66. Kg1 Kg3 67. Rg8+ Kf4 68. Ra8 Ra2 69. Kh2 a3 70. Ra4+ Ke3 71.
Rxh4 f4 72. Rh8 Rb2 73. Ra8 a2 74. h4 f3 75. Ra3+ Kd4 76. Ra4+ Kc3 77. Ra3+ Kb4 78. Ra8 f2 79.
Rb8+ Kc3 80. Rc8+ Kb3 81. Rb8+ Kc2 82. Rc8+ Kd1 83. Ra8 f1=Q 84. Rd8+ Kc1 85. Rc8+ Rc2 86.
Rxc2+ Kxc2 87. h5 a1=Q 88. Kg3 Qe5+ 89. Kh3 Qxh5+ 90. Kg3 Qg5+ 91. Kh3 Qfxg2#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 17349441883479601868 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1-0** — checkmate, 143 plies

`seed 4597470340042026716` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000244`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 e5 2. Nf3 Nc6 3. d4 exd4 4. Nxd4 Nf6 5. Nxc6 dxc6 6. Qxd8+ Kxd8 7. Nc3 Be6 8. Be2 Nd7 9.
f4 f6 10. Be3 Bb4 11. O-O-O a5 12. a3 Bxc3 13. bxc3 Kc8 14. c4 b6 15. h4 h5 16. Rhe1 Bg4 17.
Bxg4 hxg4 18. Bf2 Re8 19. h5 c5 20. Bg3 a4 21. e5 fxe5 22. fxe5 Nf8 23. Re4 Ne6 24. Rxg4 Kb7
25. Rg6 Rac8 26. Kb2 Ra8 27. Bh4 Nf4 28. Rxg7 Nxh5 29. Rh7 Nf4 30. Rdd7 Rac8 31. Rhg7 Ne6 32.
Rge7 Rxe7 33. Rxe7 Nd4 34. Bf6 Nf5 35. Rf7 Ne3 36. Kc3 Re8 37. g3 Ng4 38. Kd3 Nxe5+ 39. Bxe5
Rxe5 40. g4 Re1 41. g5 Rg1 42. Rf5 Rg3+ 43. Ke4 Rxa3 44. Kf4 Rc3 45. g6 Rxc4+ 46. Kg3 Rc3+ 47.
Kh4 Rxc2 48. g7 Rc4+ 49. Kh5 Rc1 50. g8=Q Rh1+ 51. Kg6 Rg1+ 52. Rg5 Rxg5+ 53. Kxg5 b5 54. Qd5+
Kb6 55. Kf6 b4 56. Ke6 c6 57. Qd8+ Kb5 58. Kd6 b3 59. Qb8+ Kc4 60. Kxc6 a3 61. Qf4+ Kd3 62.
Qf5+ Kd2 63. Qf2+ Kc1 64. Qxc5+ Kb2 65. Kb5 a2 66. Qd4+ Kb1 67. Qd1+ Kb2 68. Kb4 a1=Q 69. Qd2+
Kb1 70. Kxb3 Qd4 71. Qe1+ Qd1+ 72. Qxd1#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 4597470340042026716 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1-0** — checkmate, 49 plies

`seed 5337586865443150713` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000453`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 e6 6. Ndb5 Bc5 7. Nd6+ Ke7 8. e5 Bxd6 9.
Qxd6+ Ke8 10. exf6 Qxf6 11. Nb5 Rb8 12. Bf4 Qxb2 13. Rd1 Ra8 14. Nc7+ Kd8 15. Bb5 Qxb5 16. Nxb5
Ke8 17. Nc7+ Kd8 18. Nxa8 Ke8 19. Nc7+ Kd8 20. Nxe6+ Ke8 21. Nxg7+ Kd8 22. Bg5+ f6 23. Bxf6+
Ne7 24. Ne6+ Ke8 25. Qxe7#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 5337586865443150713 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1-0** — checkmate, 125 plies

`seed 2629405817628252602` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000782`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. d4 d5 2. c4 Nf6 3. cxd5 c6 4. dxc6 Nxc6 5. Nf3 e5 6. dxe5 Qxd1+ 7. Kxd1 Ng4 8. Ke1 Bc5 9. e3
Nb4 10. Bb5+ Ke7 11. Ke2 Rd8 12. h3 a6 13. Bc4 Bd7 14. hxg4 h6 15. a3 Nc2 16. Ra2 Bxg4 17. b4
Ba7 18. Rxc2 Rac8 19. Bb3 Rxc2+ 20. Bxc2 Rc8 21. Bb3 Bb8 22. Bb2 Kf8 23. Nd2 Rd8 24. Bd4 Kg8
25. Ne4 Rf8 26. Bc5 Rd8 27. Nd6 Bxd6 28. exd6 Kf8 29. Bd5 b6 30. Bxb6 Rxd6 31. Bc5 Ke7 32.
Bxd6+ Kxd6 33. Bxf7 Ke7 34. Bg6 Kf6 35. Be4 Ke6 36. Rc1 g5 37. Rc6+ Ke7 38. Rxa6 h5 39. Kd3 Bd7
40. Ne5 Bb5+ 41. Kd4 Bxa6 42. a4 Bc8 43. b5 h4 44. a5 Kf8 45. a6 Kg8 46. a7 h3 47. gxh3 Bxh3
48. a8=Q+ Kg7 49. Qd8 Be6 50. Qe7+ Kg8 51. Bh7+ Kh8 52. Bg6 Kg8 53. Qh7+ Kf8 54. Qf7+ Bxf7 55.
Bxf7 g4 56. b6 g3 57. fxg3 Kg7 58. b7 Kf6 59. b8=Q Kf5 60. Qg8 Kf6 61. Qf8 Kg5 62. Qg7+ Kf5 63.
Qg6#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 2629405817628252602 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1/2-1/2** — threefold_repetition, 133 plies

`seed 1826346386690165095` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000502`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. e4 c5 2. Nc3 Nc6 3. Nf3 Nf6 4. d4 cxd4 5. Nxd4 d6 6. Be2 g6 7. Be3 Bd7 8. Qd2 Bg7 9. f3 O-O
10. O-O-O Rc8 11. Kb1 Nxd4 12. Bxd4 Be6 13. Rhe1 Qa5 14. Nd5 Qd8 15. c4 Bxd5 16. cxd5 Nd7 17.
Bxg7 Kxg7 18. Qd4+ Kg8 19. Qxa7 Nc5 20. Qa3 Qb6 21. Rc1 Ra8 22. Qe3 Ra4 23. a3 Rfa8 24. Rc3 Kg7
25. Rec1 R4a5 26. R1c2 Ra4 27. e5 Rh4 28. h3 Na4 29. Qxb6 Nxb6 30. Rb3 Nxd5 31. exd6 exd6 32.
Bc4 Rd4 33. Bxd5 Rxd5 34. Rxb7 Rg5 35. g4 Re5 36. Rcc7 Rf8 37. b4 h5 38. Kc2 Re3 39. Rc3 Re2+
40. Kb3 Rfe8 41. gxh5 gxh5 42. a4 d5 43. a5 d4 44. Rcc7 R8e3+ 45. Kc4 Rxf3 46. Kxd4 Rf4+ 47.
Kc3 Re3+ 48. Kb2 Rf2+ 49. Rc2 Rff3 50. a6 Rb3+ 51. Kc1 Rxh3 52. Rcc7 Ra3 53. a7 Kg6 54. Kb2
Rhb3+ 55. Kc2 Rg3 56. Rc6+ Kg7 57. Kb2 Rgb3+ 58. Kc1 h4 59. Kc2 Rf3 60. Kb2 h3 61. Rc5 Kg6 62.
Rb6+ f6 63. Ra5 Rfb3+ 64. Kc2 Rc3+ 65. Kb2 Rcb3+ 66. Kc2 Rc3+ 67. Kb2
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 1826346386690165095 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

**1/2-1/2** — threefold_repetition, 134 plies

`seed 3054462775152212835` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000750`

```
[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]

1. d4 d5 2. c4 e6 3. Nf3 Nf6 4. e3 Be7 5. b3 dxc4 6. bxc4 c5 7. Nc3 cxd4 8. exd4 O-O 9. Be2 b6
10. O-O Bb7 11. Rb1 Nbd7 12. Qb3 Ne4 13. Re1 Nxc3 14. Qxc3 Qc7 15. Ba3 Rfe8 16. Bxe7 Rxe7 17.
a4 e5 18. a5 exd4 19. Nxd4 Rae8 20. Bf1 Rxe1 21. Rxe1 Rxe1 22. Qxe1 Qe5 23. Qxe5 Nxe5 24. axb6
axb6 25. f4 Nd7 26. Nb3 Kf8 27. Kf2 Ke7 28. Ke3 Nc5 29. Nxc5 bxc5 30. g3 Kd6 31. Bd3 h6 32. Be4
Bc8 33. Bd5 f6 34. Be4 Kc7 35. Bd5 Kd6 36. Be4 Bg4 37. Bc2 Bd7 38. Kd2 g5 39. Bg6 Be6 40. Bd3
gxf4 41. gxf4 Bg4 42. Ke3 Bh3 43. Ke2 Bg4+ 44. Ke3 Bd7 45. Be4 Be6 46. Bd3 Bg4 47. Bc2 Bc8 48.
Ba4 Be6 49. Bb3 Bd7 50. Bc2 Ke7 51. Kf3 Kd6 52. Kg3 Kc7 53. Kh4 Kb6 54. Kh5 Be6 55. Kxh6 Bxc4
56. h4 Be2 57. Kg6 c4 58. Kxf6 Bd3 59. Bd1 c3 60. h5 Kc5 61. f5 Be2 62. Bc2 Bxh5 63. Ke5 Bf7
64. f6 Kc6 65. Bd3 Kc5 66. Bc2 Kc6 67. Bd3 Kc5
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 3054462775152212835 --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR_w_KQkq_-_0_1)

## book

**1-0** — checkmate, 98 plies

`seed 10649976104494286963` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000801`

```
[FEN "rnbqkbnr/pp3ppp/4p3/2pp4/3P4/2PBP3/PP3PPP/RNBQK1NR b KQkq - 0 4"]

4... Nf6 5. Nd2 Be7 6. Ngf3 Nbd7 7. b3 b6 8. Bb2 Bb7 9. Qe2 O-O 10. O-O Bd6 11. c4 cxd4 12.
exd4 Qe7 13. Rfd1 Rac8 14. Ne5 Rfd8 15. a3 g6 16. Rac1 Nh5 17. g3 Bxe5 18. dxe5 Nc5 19. cxd5
Nxd3 20. Rxc8 Nhf4 21. gxf4 Nxb2 22. d6 Qh4 23. Rxd8+ Qxd8 24. Rc1 Qh4 25. Rc7 Bd5 26. f3 Qxf4
27. d7 Qg5+ 28. Qg2 Qe3+ 29. Qf2 Qg5+ 30. Kf1 Kg7 31. Qd4 Qd8 32. Rc2 Qxd7 33. Qxb2 Qe7 34. Qc3
Qg5 35. Qd4 Qh5 36. h4 Bxf3 37. Qf4 Bd1 38. Rc7 Qf5 39. Qxf5 gxf5 40. Rxa7 Bg4 41. Rb7 f4 42.
Rxb6 Kg6 43. a4 Bf5 44. a5 Bd3+ 45. Kf2 Kf5 46. a6 Kxe5 47. a7 Be4 48. Nxe4 Kxe4 49. a8=Q+ Kd4
50. Qc6 f3 51. Rb4+ Ke5 52. Qc5+ Kf6 53. Qg5#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 10649976104494286963 --fen 'rnbqkbnr/pp3ppp/4p3/2pp4/3P4/2PBP3/PP3PPP/RNBQK1NR b KQkq - 0 4' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pp3ppp/4p3/2pp4/3P4/2PBP3/PP3PPP/RNBQK1NR_b_KQkq_-_0_4)

**1-0** — checkmate, 100 plies

`seed 17144167684058451502` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000187`

```
[FEN "r1bqkbnr/pp1p1ppp/2n5/2p1p3/3PP3/2P2N2/PP3PPP/RNBQKB1R b KQkq - 0 4"]

4... exd4 5. cxd4 cxd4 6. Nxd4 Nf6 7. Nc3 Bb4 8. Nxc6 bxc6 9. Bd3 O-O 10. O-O d5 11. exd5 cxd5
12. Bg5 Bxc3 13. bxc3 h6 14. Bh4 Qd6 15. Re1 Bd7 16. Bg3 Qc5 17. Be5 Ne4 18. Bd4 Qa5 19. Bxe4
dxe4 20. Rxe4 Rfe8 21. Rf4 Re6 22. Qf3 Rae8 23. h4 f6 24. Be3 Bc6 25. Qg3 Qxc3 26. Rd1 Rxe3 27.
fxe3 Rxe3 28. Qf2 Qe5 29. Rg4 Kh7 30. Rf1 h5 31. Rc4 Re2 32. Qf5+ g6 33. Qxe5 Rxg2+ 34. Kh1
Rf2+ 35. Rxc6 Rxf1+ 36. Kg2 fxe5 37. Kxf1 Kg7 38. Ra6 Kf7 39. Rxa7+ Ke6 40. Ra6+ Kf5 41. a4 g5
42. hxg5 Kxg5 43. a5 h4 44. Re6 Kf5 45. Rh6 h3 46. a6 h2 47. Rxh2 e4 48. a7 e3 49. a8=Q e2+ 50.
Rxe2 Kg6 51. Qg8+ Kf5 52. Qf7+ Kg4 53. Rg2+ Kh4 54. Qh7#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 17144167684058451502 --fen 'r1bqkbnr/pp1p1ppp/2n5/2p1p3/3PP3/2P2N2/PP3PPP/RNBQKB1R b KQkq - 0 4' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1bqkbnr/pp1p1ppp/2n5/2p1p3/3PP3/2P2N2/PP3PPP/RNBQKB1R_b_KQkq_-_0_4)

**1/2-1/2** — threefold_repetition, 123 plies

`seed 317669806211983077` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000861`

```
[FEN "rn1qkb1r/pp3ppp/2p1pn2/3p1b2/2PP4/2NBPN2/PP3PPP/R1BQK2R b KQkq - 3 6"]

6... Bg6 7. a4 Bb4 8. O-O Nbd7 9. Qc2 O-O 10. Bxg6 hxg6 11. c5 Re8 12. a5 Bxa5 13. Rd1 Bc7 14.
h3 e5 15. Nh2 a5 16. Qb3 Rb8 17. Qc2 e4 18. Bd2 Re6 19. Ne2 g5 20. Ng3 Nf8 21. b4 axb4 22. Rdb1
Ng6 23. Rxb4 Nh4 24. Ra7 Qc8 25. Qd1 Re8 26. Rb3 g6 27. Ba5 Bxa5 28. Rxa5 Ra8 29. Rxa8 Qxa8 30.
Qb1 Re7 31. Ne2 Qa6 32. Ng3 Nf5 33. Rb6 Qa8 34. Nhf1 Nh4 35. Ne2 Kg7 36. Nc3 Nf5 37. g4 Nh4 38.
Nd2 Ne8 39. Qa2 Qxa2 40. Nxa2 f5 41. Nc3 fxg4 42. hxg4 Nf6 43. Nf1 Nxg4 44. Ng3 Rf7 45. Rb2 Kg8
46. Ncxe4 dxe4 47. Nxe4 Re7 48. Nd6 Nf6 49. Nxb7 g4 50. Nd8 Rh7 51. Nxc6 Nf3+ 52. Kf1 Ne4 53.
Rb8+ Kg7 54. Rb7+ Kh8 55. Rxh7+ Kxh7 56. Nb4 Kg7 57. c6 Nd6 58. c7 Nc8 59. Kg2 Kf6 60. Kg3 Kf5
61. Na6 Ng1 62. Kg2 Nf3 63. Nc5 Ng5 64. Kg3 Nd6 65. Kg2 Nc8 66. Kg3 Nd6 67. Kh2 Nc8
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 317669806211983077 --fen 'rn1qkb1r/pp3ppp/2p1pn2/3p1b2/2PP4/2NBPN2/PP3PPP/R1BQK2R b KQkq - 3 6' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rn1qkb1r/pp3ppp/2p1pn2/3p1b2/2PP4/2NBPN2/PP3PPP/R1BQK2R_b_KQkq_-_3_6)

**1/2-1/2** — threefold_repetition, 50 plies

`seed 11287767956172429844` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000256`

```
[FEN "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N3P1/PPPP1P1P/R1BQKBNR b KQkq - 0 3"]

3... d5 4. exd5 Nxd5 5. Bg2 Nxc3 6. bxc3 Bd6 7. Ne2 O-O 8. O-O Re8 9. d4 Nd7 10. Re1 Qf6 11.
Rb1 Nb6 12. dxe5 Rxe5 13. Nd4 Rb8 14. Rxe5 Bxe5 15. Qd3 h6 16. Ba3 Bd7 17. Bc5 Bd6 18. Bxb6
axb6 19. Rd1 Bc5 20. h4 Re8 21. Bxb7 Bg4 22. Rf1 Qd6 23. Be4 Bd7 24. Bg2 Qf6 25. Nb5 Bg4 26.
Nd4 Bd7 27. Nb3 Bd6 28. Nd4
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 11287767956172429844 --fen 'rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N3P1/PPPP1P1P/R1BQKBNR b KQkq - 0 3' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N3P1/PPPP1P1P/R1BQKBNR_b_KQkq_-_0_3)

**1/2-1/2** — threefold_repetition, 154 plies

`seed 17664257378063623849` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000467`

```
[FEN "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"]

4. Ne2 Na5 5. Bb3 Nxb3 6. axb3 d5 7. exd5 Qxd5 8. Nbc3 Qxf3 9. gxf3 Bf5 10. d3 a6 11. Bg5 Nd7
12. Be3 O-O-O 13. O-O-O f6 14. Kb1 Bh3 15. f4 exf4 16. Nxf4 Bf5 17. h4 Ne5 18. h5 Bd6 19. Ne4
Be7 20. Ng3 Bg4 21. Rde1 Nf3 22. Rc1 Rhe8 23. Nge2 Bd6 24. Nc3 Bxf4 25. Bxf4 Bf5 26. Be3 b6 27.
b4 g5 28. hxg6 hxg6 29. Rh6 Re6 30. Rch1 g5 31. Rh8 Rxh8 32. Rxh8+ Kd7 33. Kc1 Bg6 34. Rh3 Nh4
35. Kd2 Bh5 36. Ne4 Bf3 37. Nxf6+ Rxf6 38. Bxg5 Rf5 39. Bxh4 Bg2 40. Rh2 Bf3 41. Bg3 Rh5 42.
Rxh5 Bxh5 43. f4 Ke6 44. f5+ Kxf5 45. Bxc7 b5 46. Ke3 Bd1 47. Kd2 Bf3 48. Bh2 Bc6 49. Kc3 Bf3
50. Kd4 Ke6 51. Kc5 Kd7 52. Kb6 Be2 53. Kxa6 Kc6 54. Bb8 Bf1 55. Ba7 Be2 56. Bb6 Bf1 57. b3 Be2
58. Be3 Bf1 59. Bb6 Be2 60. d4 Kd5 61. Ba7 Kc6 62. c4 bxc4 63. b5+ Kd5 64. bxc4+ Bxc4 65. Ka5
Bb3 66. Bc5 Bc4 67. Bb6 Ke6 68. Kb4 Bd5 69. Kc5 Bf3 70. Ba5 Bg2 71. b6 Bd5 72. Bc3 Be4 73. Ba1
Ba8 74. Bb2 Bb7 75. Ba3 Ba8 76. Kb5 Bb7 77. Bc5 Kd7 78. Kc4 Ke6 79. Kb5 Kd5 80. Kb4 Ke6
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 17664257378063623849 --fen 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR_w_KQkq_-_4_4)

**1/2-1/2** — threefold_repetition, 145 plies

`seed 12635671946112545669` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000826`

```
[FEN "rnbqkb1r/pp3ppp/2p2n2/3p4/2PP4/5N2/PP3PPP/RNBQKB1R w KQkq - 0 6"]

6. Nc3 Be7 7. Bd3 O-O 8. O-O dxc4 9. Bxc4 Nbd7 10. Bb3 a5 11. Re1 Nb6 12. h3 Nbd5 13. Bc2 Bd6
14. Bg5 Be6 15. Qd3 Nb4 16. Qd2 h6 17. Bh4 Bf4 18. Qd1 g5 19. Ne2 Bd6 20. Bg3 Nxc2 21. Qxc2 Nd5
22. Ne5 Re8 23. Nc3 Bf8 24. Rad1 Bg7 25. a3 Ne7 26. Ne4 Nf5 27. Bh2 Bd5 28. Nc3 Nh4 29. f3 f6
30. Nxd5 fxe5 31. Nc3 exd4 32. Rxe8+ Qxe8 33. Ne4 Qe6 34. Re1 a4 35. Bg3 Qd5 36. Qd3 Nf5 37.
Bf2 c5 38. Ng3 Nxg3 39. Bxg3 c4 40. Qg6 Qf7 41. Qd6 Re8 42. Rxe8+ Qxe8 43. Qd5+ Qf7 44. Qd8+
Kh7 45. Qd6 c3 46. bxc3 dxc3 47. Qd3+ Qg6 48. Qd7 c2 49. Qd2 Bb2 50. Qd7+ Bg7 51. Qd2 b5 52.
Kh2 Qc6 53. Qc1 Qc4 54. Be1 Be5+ 55. Bg3 Bxg3+ 56. Kxg3 Qd3 57. Qe1 Qxa3 58. Qe4+ Kg7 59. Qxc2
Qd6+ 60. Kf2 a3 61. Qc3+ Kh7 62. Qb3 b4 63. Qf7+ Kh8 64. Ke2 Qe5+ 65. Kd3 Qc3+ 66. Ke4 a2 67.
Qxa2 b3 68. Qa8+ Kg7 69. Qb7+ Kf6 70. Qa6+ Kf7 71. Qa7+ Kf8 72. Qb8+ Ke7 73. Qb7+ Kf8 74. Kf5
Qd3+ 75. Kf6 Qd6+ 76. Kf5 Qd3+ 77. Kf6 Qd6+ 78. Kf5
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 12635671946112545669 --fen 'rnbqkb1r/pp3ppp/2p2n2/3p4/2PP4/5N2/PP3PPP/RNBQKB1R w KQkq - 0 6' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkb1r/pp3ppp/2p2n2/3p4/2PP4/5N2/PP3PPP/RNBQKB1R_w_KQkq_-_0_6)

**1-0** — checkmate, 100 plies

`seed 8036381349872193540` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000067`

```
[FEN "r1bqk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2QK2R b KQkq - 1 6"]

6... Bb4 7. Bd2 Na5 8. Bb5+ c6 9. Ba4 O-O 10. O-O Bxc3 11. Bxc3 b5 12. Bb3 Nxb3 13. axb3 a5 14.
Qd2 c5 15. Bxa5 Qe7 16. b4 Bg4 17. bxc5 dxc5 18. Qe3 Bxf3 19. Qxf3 Qd6 20. Bc3 b4 21. Bd2 h6
22. h3 Rfe8 23. Qe2 Nd7 24. Be3 Nf8 25. Qg4 Kh7 26. Rxa8 Rxa8 27. Qf5+ Kg8 28. f4 exf4 29. Bxc5
Qc7 30. Bxb4 Ne6 31. Bc3 Qb6+ 32. Kh2 Qe3 33. Qd5 Ra6 34. Qf5 Qe2 35. Kg1 Qxc2 36. d4 Ng5 37.
Qc8+ Kh7 38. Qxa6 Qxe4 39. Qd6 f3 40. Qg3 fxg2 41. Qxg2 Qe3+ 42. Kh1 Nxh3 43. Rxf7 Qg5 44. Qxg5
hxg5 45. d5 Kg6 46. Rxg7+ Kf5 47. d6 Ke6 48. d7 Nf2+ 49. Kg2 Ne4 50. d8=Q Nxc3 51. Re7+ Kf6 52.
Qf8+ Kg6 53. Rg7+ Kh5 54. Qh8+ Kg4 55. Rf7 Ne4 56. Qh3#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 8036381349872193540 --fen 'r1bqk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2QK2R b KQkq - 1 6' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1bqk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2QK2R_b_KQkq_-_1_6)

**1/2-1/2** — threefold_repetition, 67 plies

`seed 3820341833905020357` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000277`

```
[FEN "r1bqkbnr/pppppppp/2n5/1B6/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2"]

2... d5 3. e5 Bd7 4. d4 Nxe5 5. Be2 Ng6 6. Nf3 e6 7. h4 Bd6 8. h5 Nf4 9. Bf1 Nf6 10. Ne5 g5 11.
g3 Ne4 12. gxf4 c5 13. Nc3 Nxc3 14. bxc3 h6 15. Nxd7 Qxd7 16. fxg5 hxg5 17. Bxg5 Qc7 18. h6 a6
19. Bf6 Rh7 20. Qd3 cxd4 21. Qxh7 Qxc3+ 22. Ke2 Be7 23. Bxe7 Kxe7 24. Qd3 Qxa1 25. Qa3+ Kd7 26.
Bg2 d3+ 27. Kxd3 Qe5 28. h7 Qf5+ 29. Ke3 Qe5+ 30. Kd2 Qd4+ 31. Ke2 Rh8 32. Qf3 Qc4+ 33. Kd1
Qd4+ 34. Ke2 Qc4+ 35. Kd1 Qd4+
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 3820341833905020357 --fen 'r1bqkbnr/pppppppp/2n5/1B6/4P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1bqkbnr/pppppppp/2n5/1B6/4P3/8/PPPP1PPP/RNBQK1NR_b_KQkq_-_2_2)

**1/2-1/2** — threefold_repetition, 119 plies

`seed 9271918869534182246` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000705`

```
[FEN "rnbqkbnr/pp1p1ppp/8/4p3/3PP3/8/PP3PPP/RNBQKBNR b KQkq - 0 4"]

4... exd4 5. Nf3 Nc6 6. Bc4 Bb4+ 7. Bd2 Nf6 8. Bxb4 Nxb4 9. O-O d5 10. Bb5+ Bd7 11. Bxd7+ Qxd7
12. e5 Ne4 13. Nxd4 Nc6 14. Nc3 O-O 15. Nxc6 Nxc3 16. bxc3 Qxc6 17. Qd4 b6 18. Rad1 Rfd8 19.
Rd3 Rac8 20. f4 Qc4 21. f5 Re8 22. Qxd5 h6 23. Qxc4 Rxc4 24. Re3 Rc5 25. e6 Kf8 26. a4 a6 27.
h3 b5 28. axb5 axb5 29. Rf4 Rd5 30. Kh2 Rc5 31. h4 h5 32. Kg3 f6 33. Rd3 Ke7 34. Rd7+ Kf8 35.
Kf3 Ra8 36. g4 Rxc3+ 37. Kg2 Rc2+ 38. Kg3 Rc3+ 39. Rf3 b4 40. Rxc3 bxc3 41. Rc7 Ra2 42. Rc8+
Ke7 43. Rxc3 hxg4 44. Rc7+ Kf8 45. Rc8+ Ke7 46. Kxg4 Ra1 47. Rc7+ Kf8 48. Rd7 Rg1+ 49. Kh5 Rg2
50. Rf7+ Kg8 51. Ra7 Kf8 52. Ra8+ Ke7 53. Rg8 Rg1 54. Rb8 Rg3 55. Rb7+ Kf8 56. Rc7 Rg1 57. Rc8+
Ke7 58. Rb8 Rg3 59. Rb7+ Kf8 60. Ra7 Rg1 61. Rd7 Rg2 62. Rd8+ Ke7 63. Rg8 Rg1
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 9271918869534182246 --fen 'rnbqkbnr/pp1p1ppp/8/4p3/3PP3/8/PP3PPP/RNBQKBNR b KQkq - 0 4' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkbnr/pp1p1ppp/8/4p3/3PP3/8/PP3PPP/RNBQKBNR_b_KQkq_-_0_4)

**1-0** — checkmate, 110 plies

`seed 1253587804697920911` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000737`

```
[FEN "rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/2P5/PP2PPPP/RN1QKBNR b KQkq - 2 3"]

3... Nbd7 4. e3 Ne4 5. Bh4 g5 6. Bg3 h5 7. a3 h4 8. Nd2 hxg3 9. Nxe4 dxe4 10. fxg3 Bh6 11. Qb3
O-O 12. g4 Nf6 13. h3 Qd6 14. Qc2 Nd5 15. Kd2 Qg3 16. Qxe4 Nf6 17. Qf3 Qd6 18. Bd3 c5 19. Ne2
Be6 20. Ng3 Rfd8 21. Kc2 Bg7 22. Rhf1 Rac8 23. Rae1 cxd4 24. exd4 Qb6 25. Nf5 Nd5 26. Nxg7 Kxg7
27. Qe4 Kf8 28. Qxe6 Qxe6 29. Rxe6 Nf4 30. Re5 f6 31. Re4 Nxg2 32. Rf2 Nf4 33. h4 Nxd3 34. Kxd3
gxh4 35. Rh2 Kf7 36. Rxh4 e5 37. Rh7+ Kg6 38. Rxb7 a5 39. Rb5 a4 40. d5 Ra8 41. c4 Ra6 42. Kc3
Raa8 43. Kb4 Kg5 44. Rb6 Kg6 45. d6 Rdb8 46. Rxb8 Rxb8+ 47. Kxa4 Rxb2 48. c5 Kf7 49. c6 Ke6 50.
d7 Ke7 51. Ka5 Rd2 52. Kb6 Rb2+ 53. Kc7 Rd2 54. Rb4 Rd6 55. Rb8 Rxc6+ 56. Kxc6 e4 57. d8=Q+ Ke6
58. Qe8#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 1253587804697920911 --fen 'rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/2P5/PP2PPPP/RN1QKBNR b KQkq - 2 3' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkb1r/ppp1pppp/5n2/3p2B1/3P4/2P5/PP2PPPP/RN1QKBNR_b_KQkq_-_2_3)

## archive

**1/2-1/2** — insufficient_material, 106 plies

`seed 13590854816084957743` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000860`

```
[FEN "r1bqk2r/1p3ppp/p1nb1n2/1p6/4P3/2N1BN2/PPP2PPP/R2QK2R w KQkq - 0 1"]

1. Qd3 Be7 2. e5 Qxd3 3. cxd3 Ng4 4. Bf4 Nb4 5. O-O Nxd3 6. Nd5 Nxf4 7. Nc7+ Kf8 8. Nxa8 Be6 9.
Rfd1 g5 10. h3 Nh6 11. Nc7 Nf5 12. Rac1 Ne2+ 13. Kf1 Nxc1 14. Nxe6+ fxe6 15. Rxc1 h5 16. Rc8+
Kg7 17. Rc7 g4 18. Ng5 Kg6 19. Nxe6 Re8 20. Rxb7 Rc8 21. Nf4+ Kg5 22. g3 Rc1+ 23. Kg2 Rc2 24.
Rb6 Nxg3 25. Ne6+ Kh4 26. Nd4 Rxf2+ 27. Kxf2 Bc5 28. Rd6 Ne4+ 29. Ke3 Nxd6 30. exd6 Bxd6 31.
hxg4 Kxg4 32. Nf3 h4 33. Nxh4 Kxh4 34. Kd4 Kg4 35. Kd5 Be7 36. Kc6 Kf4 37. Kb6 Ke3 38. Kxa6 b4
39. a4 b3 40. Kb5 Kd4 41. a5 Bc5 42. a6 Kd5 43. a7 Bxa7 44. Kb4 Kd4 45. Kxb3 Bc5 46. Kc2 Bb4
47. Kb3 Kc5 48. Ka4 Be1 49. b4+ Kc4 50. Ka5 Bf2 51. b5 Kd5 52. b6 Kc6 53. Ka6 Bxb6
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 13590854816084957743 --fen 'r1bqk2r/1p3ppp/p1nb1n2/1p6/4P3/2N1BN2/PPP2PPP/R2QK2R w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1bqk2r/1p3ppp/p1nb1n2/1p6/4P3/2N1BN2/PPP2PPP/R2QK2R_w_KQkq_-_0_1)

**1-0** — checkmate, 181 plies

`seed 2621727037773359101` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000171`

```
[FEN "rn1qk2r/ppp2ppp/3bp3/3n1b2/3P4/5N1P/PPP1BPP1/RNBQ1RK1 w kq - 0 1"]

1. c4 Nf4 2. c5 Nxe2+ 3. Qxe2 Be7 4. Qb5+ Nd7 5. Qxb7 Qc8 6. Qxc8+ Rxc8 7. Nc3 h5 8. Bf4 Nf6 9.
b4 h4 10. Rfe1 Bg6 11. Ne5 Bf5 12. Nc6 Kd7 13. b5 Bf8 14. Nxa7 Ra8 15. Nc6 g5 16. Bxg5 Bg7 17.
Ne5+ Kc8 18. Nxf7 Rg8 19. c6 Ra3 20. Rac1 Kb8 21. b6 cxb6 22. Nb5 Ne4 23. Bf4+ Ka8 24. Nfd6
Bxd4 25. Nxd4 Bxh3 26. Rxe4 Rxg2+ 27. Kh1 Rg8 28. c7 Bg2+ 29. Kg1 Bxe4+ 30. Kf1 Bb7 31. c8=Q+
Rxc8 32. Nxc8 Rh3 33. Nxb6+ Ka7 34. Nd7 Rh1+ 35. Ke2 Rxc1 36. Bxc1 h3 37. Bf4 Ka8 38. Bh2 Ba6+
39. Ke3 Bb7 40. Nxe6 Bg2 41. Nec5 Bb7 42. Nb6+ Ka7 43. Ncd7 Ka6 44. Kd4 Kb5 45. Nc4 Bf3 46. Nc5
Be2 47. Nd2 Kb6 48. Nc4+ Kb5 49. Nd6+ Kb6 50. Bg3 Bd1 51. Nde4 Ka5 52. Kc3 Kb6 53. Kb4 Kc6 54.
Nc3 Bc2 55. Nb3 Kd7 56. Nd4 Bg6 57. Kc5 Be8 58. a4 Kc8 59. a5 Kb7 60. Nd5 Bd7 61. a6+ Kxa6 62.
Nb6 Be8 63. Nb3 Bf7 64. Nd5 Kb7 65. Nd4 Bxd5 66. Kxd5 Kc8 67. Kc6 Kd8 68. Kd6 Kc8 69. Nc6 Kb7
70. Kc5 Ka8 71. Bd6 h2 72. Bxh2 Kb7 73. Kb5 Kc8 74. Kb6 Kd7 75. Nd4 Ke7 76. Kc5 Kf6 77. Kd6 Kg5
78. Ke5 Kh6 79. f4 Kg6 80. Ke6 Kh5 81. f5 Kg4 82. f6 Kh3 83. Bf4 Kg2 84. f7 Kf1 85. f8=Q Kf2
86. Ke5 Kg2 87. Qf5 Kf1 88. Qc2 Kg1 89. Ne2+ Kf2 90. Ng3+ Kf3 91. Qe2#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 2621727037773359101 --fen 'rn1qk2r/ppp2ppp/3bp3/3n1b2/3P4/5N1P/PPP1BPP1/RNBQ1RK1 w kq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rn1qk2r/ppp2ppp/3bp3/3n1b2/3P4/5N1P/PPP1BPP1/RNBQ1RK1_w_kq_-_0_1)

**0-1** — checkmate, 54 plies

`seed 7945266456323018428` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000103`

```
[FEN "4r3/1q2kp2/b2p2pQ/1pn1r2N/4P3/2P4P/2B3P1/R3R1K1 w - - 0 1"]

1. Nf4 Qc6 2. Nd5+ Kd8 3. Rf1 f5 4. Nb4 Qb6 5. exf5 gxf5 6. Rxf5 Rxf5 7. Bxf5 Bb7 8. Qf6+ Re7
9. Qf8+ Re8 10. Qf7 Re7 11. Qg8+ Re8 12. Qh7 Qc7 13. Qh4+ Qe7 14. Qd4 Rf8 15. Bg4 Nd7 16. Ra7
Qe1+ 17. Kh2 Rf1 18. Qh8+ Nf8 19. Nc6+ Bxc6 20. Qxf8+ Rxf8 21. Rg7 Qe5+ 22. Kg1 Qe1+ 23. Kh2
Rf2 24. Rg8+ Kc7 25. Rg7+ Kb6 26. Kg3 Rxg2+ 27. Kf4 Qe5#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 7945266456323018428 --fen '4r3/1q2kp2/b2p2pQ/1pn1r2N/4P3/2P4P/2B3P1/R3R1K1 w - - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/4r3/1q2kp2/b2p2pQ/1pn1r2N/4P3/2P4P/2B3P1/R3R1K1_w_-_-_0_1)

**1-0** — checkmate, 99 plies

`seed 13893289343309791649` · `generation-visitfloor-20260820-round-000001-worker-0003-invocation-0001-game-00000350`

```
[FEN "r1b1k2r/ppp1ppbp/2n2np1/q7/8/2NP1NP1/PPP2PBP/R1BQK2R w KQkq - 0 1"]

1. O-O O-O 2. a3 Nd5 3. Nxd5 Qxd5 4. Rb1 h5 5. a4 Qa2 6. Bd2 Qxa4 7. Ra1 Qg4 8. Bc3 Rd8 9. Qd2
Qf5 10. Rfe1 a5 11. Ng5 f6 12. Bxc6 e5 13. Be4 Qxg5 14. Qxg5 fxg5 15. Bxa5 Rf8 16. Bxc7 Rxa1
17. Rxa1 Kf7 18. Ra5 g4 19. Rb5 Re8 20. Bxb7 Re7 21. Bxc8 Rxc7 22. Rb7 Rxb7 23. Bxb7 e4 24.
Bd5+ Kf8 25. Bxe4 Bxb2 26. Bxg6 h4 27. gxh4 Kg7 28. h5 Kh6 29. c4 Bd4 30. Kg2 Kg5 31. f3 gxf3+
32. Kxf3 Kf6 33. Ke4 Bg1 34. h3 Bf2 35. d4 Ke6 36. c5 Kf6 37. Kd5 Be3 38. c6 Ke7 39. Ke5 Bf2
40. d5 Bg3+ 41. Kf5 Kd6 42. h6 Bh4 43. h7 Be7 44. h8=Q Kxd5 45. c7 Kc4 46. c8=Q+ Bc5 47. Bf7+
Kb4 48. Qb2+ Ka5 49. Qxc5+ Ka4 50. Qca3#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 13893289343309791649 --fen 'r1b1k2r/ppp1ppbp/2n2np1/q7/8/2NP1NP1/PPP2PBP/R1BQK2R w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/r1b1k2r/ppp1ppbp/2n2np1/q7/8/2NP1NP1/PPP2PBP/R1BQK2R_w_KQkq_-_0_1)

**1/2-1/2** — threefold_repetition, 106 plies

`seed 9643704804445734752` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000512`

```
[FEN "rnbqkb1r/pp3ppp/2P2n2/1B6/5p2/5N2/PPPP2PP/RNBQK2R b KQkq - 0 1"]

1... bxc6 2. Bc4 Bd6 3. Qe2+ Be7 4. d4 O-O 5. O-O Bd6 6. Ne5 Bxe5 7. dxe5 Nd5 8. Bxf4 Qb6+ 9.
Kh1 Be6 10. Nd2 Nd7 11. b3 Nxf4 12. Rxf4 Rae8 13. Bd3 g6 14. Nc4 Bxc4 15. Rxc4 Nxe5 16. Re4 Qc5
17. Re1 Nxd3 18. Qxd3 Rxe4 19. Rxe4 Qa3 20. h3 Qxa2 21. Qc3 Qa6 22. Qf6 Qa5 23. Kh2 Qd8 24.
Qxc6 Qb8+ 25. Kg1 Rc8 26. Qd7 Rd8 27. Qe7 a5 28. Kh1 h6 29. Re1 Qd6 30. Qxd6 Rxd6 31. Ra1 Rd5
32. c4 Rd3 33. Rxa5 Rxb3 34. c5 Rc3 35. Kh2 Kf8 36. h4 Ke7 37. Ra7+ Ke6 38. Ra6+ Kd7 39. h5
gxh5 40. Rxh6 Rxc5 41. Rf6 Ke7 42. Ra6 Rb5 43. Ra7+ Kf6 44. Ra6+ Kg5 45. Ra8 Rb3 46. Rg8+ Kf5
47. Rh8 Kg6 48. Rg8+ Kh6 49. g3 Rb2+ 50. Kh3 Rb1 51. Rh8+ Kg6 52. Rg8+ Kh6 53. Rh8+ Kg6 54.
Rg8+
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 9643704804445734752 --fen 'rnbqkb1r/pp3ppp/2P2n2/1B6/5p2/5N2/PPPP2PP/RNBQK2R b KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkb1r/pp3ppp/2P2n2/1B6/5p2/5N2/PPPP2PP/RNBQK2R_b_KQkq_-_0_1)

**1/2-1/2** — threefold_repetition, 71 plies

`seed 3795853539798652930` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000817`

```
[FEN "kq6/pp6/8/8/8/8/8/KQ6 w - - 0 1"]

1. Qe4 Qd8 2. Kb2 a6 3. Kb3 Qb6+ 4. Ka2 Ka7 5. Qh7 Qb4 6. Qh8 a5 7. Qh6 a4 8. Qe3+ b6 9. Qd3
Qc5 10. Kb1 Qg1+ 11. Kc2 Qg2+ 12. Kc1 Qc6+ 13. Kb2 b5 14. Kb1 Qc5 15. Qh7+ Kb6 16. Qh8 Qf5+ 17.
Kb2 Qf2+ 18. Kb1 Qe1+ 19. Kc2 Qe2+ 20. Kc1 Qc4+ 21. Kb2 Qb3+ 22. Ka1 Qa3+ 23. Kb1 Qd3+ 24. Ka1
Qd6 25. Qg8 a3 26. Ka2 b4 27. Kb3 Qd3+ 28. Kxb4 a2 29. Qxa2 Qd6+ 30. Kb3 Qd5+ 31. Ka3 Qc5+ 32.
Kb2 Qe5+ 33. Kb1 Qe1+ 34. Kb2 Qe5+ 35. Kb1 Qe1+ 36. Kb2
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 3795853539798652930 --fen 'kq6/pp6/8/8/8/8/8/KQ6 w - - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/kq6/pp6/8/8/8/8/8/KQ6_w_-_-_0_1)

**0-1** — checkmate, 41 plies

`seed 7992089784875032001` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000336`

```
[FEN "2r2rk1/1p2bppp/8/4Pb2/1q1p1R2/8/3BQ2P/R2N2K1 b - - 0 1"]

1... Qb3 2. Rxf5 Rc2 3. Rf3 Qd5 4. Ra5 Qxa5 5. Rd3 Qa2 6. Nf2 Qe6 7. Qd1 Ra2 8. Rb3 Qg6+ 9. Bg5
Qxg5+ 10. Qg4 Ra1+ 11. Kg2 Qxe5 12. Rg3 Ra2 13. Kf1 Ra1+ 14. Nd1 Bb4 15. Kg2 Ra2+ 16. Nf2 f5
17. Qg5 Qe4+ 18. Kg1 Qe1+ 19. Kg2 Rxf2+ 20. Kh3 Qe5 21. Rb3 Qxh2#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 7992089784875032001 --fen '2r2rk1/1p2bppp/8/4Pb2/1q1p1R2/8/3BQ2P/R2N2K1 b - - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/2r2rk1/1p2bppp/8/4Pb2/1q1p1R2/8/3BQ2P/R2N2K1_b_-_-_0_1)

**1-0** — checkmate, 40 plies

`seed 8608957559821739777` · `generation-visitfloor-20260820-round-000001-worker-0000-invocation-0001-game-00000855`

```
[FEN "rn1qkb1r/pp2pppp/2pp4/3nP1N1/3P2b1/8/PPP1BPPP/RNBQK2R b KQkq - 0 1"]

1... Bxe2 2. Qxe2 h6 3. Nf3 e6 4. O-O dxe5 5. dxe5 Nd7 6. c4 Ne7 7. Nc3 Qc7 8. Rd1 Ng6 9. h4 a5
10. h5 Ne7 11. Ne4 Nf5 12. Bf4 g5 13. hxg6 fxg6 14. Nd6+ Ke7 15. Nxf5+ gxf5 16. Nd4 Kf7 17.
Qh5+ Ke7 18. Nxe6 Kxe6 19. Qg6+ Ke7 20. e6 Qxf4 21. Rxd7#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 8608957559821739777 --fen 'rn1qkb1r/pp2pppp/2pp4/3nP1N1/3P2b1/8/PPP1BPPP/RNBQK2R b KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rn1qkb1r/pp2pppp/2pp4/3nP1N1/3P2b1/8/PPP1BPPP/RNBQK2R_b_KQkq_-_0_1)

**1/2-1/2** — threefold_repetition, 117 plies

`seed 6982294511850118467` · `generation-visitfloor-20260820-round-000001-worker-0002-invocation-0001-game-00000186`

```
[FEN "rnbqkb1r/pp3ppp/2p2n2/3p4/3Pp3/4P3/PPPNBPPP/RNBQK2R w KQkq - 0 1"]

1. c4 Bd6 2. Nc3 O-O 3. Qb3 Re8 4. cxd5 cxd5 5. Nxd5 Nxd5 6. Qxd5 Nc6 7. Qc4 Qg5 8. Bf1 Be6 9.
Qa4 Bd5 10. h4 Qg6 11. h5 Qe6 12. Bb5 a6 13. Bxc6 Bxc6 14. Qd1 Bb5 15. Rh4 Rac8 16. Nb1 Qf6 17.
Rg4 Qf5 18. Nc3 Rxc3 19. bxc3 Bd3 20. h6 g5 21. a4 Re6 22. f4 Rxh6 23. Kf2 Rg6 24. Ba3 gxf4 25.
Rxg6+ hxg6 26. Bxd6 fxe3+ 27. Kxe3 Qg5+ 28. Kf2 Qf6+ 29. Kg1 Qxd6 30. Qd2 Qg3 31. Re1 f5 32.
Re3 Qd6 33. Rh3 f4 34. Rh4 g5 35. Rg4 Qf6 36. Qa2+ Kg7 37. Qd5 Kh6 38. Qxb7 e3 39. Qd5 Be2 40.
Rxg5 Qxg5 41. Qe6+ Kg7 42. Qd7+ Kh6 43. Qe6+ Qg6 44. Qe5 Qg3 45. Qh8+ Kg5 46. Qg8+ Kh4 47. Qd8+
Qg5 48. Qh8+ Bh5 49. Qc8 Qg3 50. Qd8+ Kg4 51. Qg8+ Kf5 52. Qd5+ Kg4 53. Qg8+ Kf5 54. Qd5+ Kf6
55. Qe5+ Kg6 56. Qe8+ Kg7 57. Qe5+ Kg6 58. Qe6+ Kh7 59. Qe5
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 6982294511850118467 --fen 'rnbqkb1r/pp3ppp/2p2n2/3p4/3Pp3/4P3/PPPNBPPP/RNBQK2R w KQkq - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/rnbqkb1r/pp3ppp/2p2n2/3p4/3Pp3/4P3/PPPNBPPP/RNBQK2R_w_KQkq_-_0_1)

**1-0** — checkmate, 41 plies

`seed 10195479559262659399` · `generation-visitfloor-20260820-round-000001-worker-0001-invocation-0001-game-00000561`

```
[FEN "2r2rk1/p3p1bp/1qp2pp1/2RpP3/3P2b1/1P2BN2/P2Q1PPP/2R3K1 w - - 0 1"]

1. exf6 exf6 2. Ne1 Rfe8 3. h3 Bd7 4. b4 Qb7 5. Nd3 Bf8 6. Ra5 Bf5 7. Nc5 Bxc5 8. Raxc5 Re4 9.
b5 cxb5 10. Rxc8+ Bxc8 11. Qa5 Bd7 12. Rc7 Qc6 13. Rxc6 Bxc6 14. Qd8+ Kg7 15. Qc7+ Kf8 16. Qxc6
Re7 17. Qxf6+ Rf7 18. Qh8+ Ke7 19. Bg5+ Ke6 20. Qe8+ Kf5 21. Qe5#
```

<details><summary>replay locally</summary>

```sh
uv run scripts/play_self_play_games.py $C --games 1 --simulations 200 \
  --seed 10195479559262659399 --fen '2r2rk1/p3p1bp/1qp2pp1/2RpP3/3P2b1/1P2BN2/P2Q1PPP/2R3K1 w - - 0 1' --choices 3
```

</details>

[open on lichess](https://lichess.org/analysis/2r2rk1/p3p1bp/1qp2pp1/2RpP3/3P2b1/1P2BN2/P2Q1PPP/2R3K1_w_-_-_0_1)

## other

_no games from this source_

