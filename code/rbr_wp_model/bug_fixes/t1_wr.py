import polars as pl
import os
from tqdm import tqdm
from collections import defaultdict


MAPS = [
    "ancient", "anubis", "inferno", "mirage", "nuke",
    "overpass", "vertigo", "dust2", "train", "cache"
]


DATA_PATH = "/home/lz80/rdf/sp161/shared/cs2_saving/parsed"
counts = defaultdict(int)
wins = defaultdict(int)

for game_file in tqdm(sorted(os.listdir(f"{DATA_PATH}/events"))):
    if not game_file.endswith(".csv"):
        continue
    game = game_file[:-4]
    game_lower = game.lower()
    map_played = next((m for m in MAPS if m in game_lower), "unknown")
    a = pl.read_csv(f"{DATA_PATH}/rounds/{game}.csv")
    a = a.filter(pl.col("round_num") <= 24)
    a = a.with_columns(
        half = pl.when(pl.col("round_num") <= 12).then(1).otherwise(2)
        ).with_columns(
        t1_won_round=(
            ((pl.col("winner") == "ct") & pl.col("half") == 1) |
            ((pl.col("winner") == "t") & (pl.col("half") == 2))
        )
    )

    round_counts = a.select(
        t1_rounds=pl.col("t1_won_round").sum(),
        t2_rounds=24 - pl.col("t1_won_round").sum(),
    )

    winner = round_counts.select("t1_rounds").item() > round_counts.select("t2_rounds").item()
    wins[map_played] += winner
    counts[map_played] += 1
winrates = {}

for m in MAPS:
    if counts[m] == 0:
        winrates[m] = 0
    else:
        winrates[m] = wins[m] / counts[m]
print(winrates)