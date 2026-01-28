# Training Data

Output from `train_save.py` - Tic Tac Toe Q-learning self-play (100k episodes).

## Files

| File | Description |
|------|-------------|
| `episode_results.json` | Winner per episode (0=draw, 1=X, 2=O) |
| `snapshots.json.gz` | 200 snapshots every 500 eps: Q-tables + demo games |
| `evaluation.json` | Final greedy eval (10k games) |

## Snapshot Format

```python
{
  "episode": 500,
  "epsilon": 0.975,
  "q_table_x": {"(0,0,0,...)": [q0, q1, ..., q8]},
  "q_table_o": {"(0,0,0,...)": [q0, q1, ..., q8]},
  "demo_game": [
    {"board": [0,0,0,0,0,0,0,0,0], "player": 1, "action": 4},
    ...
    {"board": [...], "result": "draw"|"win", "winner": 0|1|2}
  ]
}
```

## Board

```
0|1|2
-+-+-
3|4|5
-+-+-
6|7|8
```
Values: 0=empty, 1=X, 2=O

## Load

```python
import gzip, json
with gzip.open("snapshots.json.gz", "rt") as f:
    snapshots = json.load(f)  # list of 200 snapshots
```
