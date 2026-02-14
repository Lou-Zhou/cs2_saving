import os
import re
from tqdm import tqdm
import traceback
import polars as pl
import polars.selectors as cs
import numpy as np

DATA_PATH = "/home/lz80/rdf/sp161/shared/cs2_saving/parsed"

t2_rifle = ["famas", "galil ar", "ssg 08"]
t1_rifle = ["ak-47", "m4a4", "m4a1-s", "aug", "sg 553"]

smg = ["mp9", "mac-10", "mp5-sd", "mp7", "pp-bizon", "p90", "ump-45"]
heavy = ['mag-7', 'nova', 'sawed-off', 'xm1014', 'm249', 'negev']
t1_pistol = ["five-seven", "cz75-auto", 'dual berettas', 'p250', 'r8 revolver', 'tec-9', 'desert eagle']
grenade = ['high explosive grenade', 'flashbang', 'smoke grenade', 'decoy grenade', 'molotov', 'incendiary grenade']

MAPS = [
    "ancient", "anubis", "inferno", "mirage", "nuke",
    "overpass", "vertigo", "dust2", "train", "cache"
]

LOSS_BONUS_STEPS = np.array([1400, 1900, 2400, 2900, 3400], dtype=np.float64)


def compute_loss_bonus(round_outcomes: np.ndarray) -> np.ndarray:
    """
    Compute round-start loss bonus from prior results.
    Counter is capped [0, 4], increments on a loss, decrements by one on a win.
    """
    counter = 0
    loss_bonus = np.empty(round_outcomes.shape[0], dtype=np.float64)
    for i, won_round in enumerate(round_outcomes):
        loss_bonus[i] = LOSS_BONUS_STEPS[counter]
        if won_round:
            counter = max(counter - 1, 0)
        else:
            counter = min(counter + 1, 4)
    return loss_bonus


gun_cost = {
    "awp":4750,
    # t2_rifle
    "famas": 1950,
    "galil ar": 1800,
    "ssg 08": 1700,

    # t1_rifle
    "ak-47": 2700,
    "m4a4": 2900,
    "m4a1-s": 2900,
    "aug": 3300,
    "sg 553": 3000,

    # smg
    "mp9": 1250,
    "mac-10": 1050,
    "mp5-sd": 1400,
    "mp7": 1400,
    "pp-bizon": 1300,
    "p90": 2350,
    "ump-45": 1200,

    # heavy (shotguns + lmgs)
    "mag-7": 1300,
    "nova": 1050,
    "sawed-off": 1100,
    "xm1014": 2000,
    "m249": 5200,
    "negev": 1700,

    # t1_pistol
    "five-seven": 500,
    "cz75-auto": 500,
    "dual berettas": 300,
    "p250": 300,
    "r8 revolver": 600,
    "tec-9": 500,
    "desert eagle": 700,
}
gun_cost.update({
    "decoy grenade": 50,
    "flashbang": 200,
    "high explosive grenade": 300,
    "smoke grenade": 300,
    "molotov": 400,            # T
    "incendiary grenade": 500, # CT
})




dfs = []
seen_games = set()
for game_file in tqdm(sorted(os.listdir(f"{DATA_PATH}/events"))):
    if not game_file.endswith(".csv"):
        continue
    game = game_file[:-4]
    # Deduplicate alternate parses of the same demo:
    # - optional ".dem" token in stem
    # - optional trailing numeric suffix (e.g. "-1", "-2")
    game_norm = re.sub(r"\.dem$", "", game)
    game_norm = re.sub(r"-\d+$", "", game_norm)
    if game_norm in seen_games:
        continue
    seen_games.add(game_norm)
    try:
        events = pl.read_csv(f"{DATA_PATH}/events/{game}.csv")
        rounds = pl.read_csv(f"{DATA_PATH}/rounds/{game}.csv")
        ticks = pl.read_json(f"{DATA_PATH}/ticks/{game}.json")
#/home/lz80/rdf/sp161/shared/cs2_saving/parsed/events/themongolz-aurora-m2-mirage.dem.csv
        game_lower = game.lower()

        map_played = next((m for m in MAPS if m in game_lower), "unknown")

        first_tick = (
            ticks.filter(pl.col("round_num") == 1)
            .filter(pl.col("tick") == pl.col("tick").min())

            )

                # Build stable player->team lookup from first-half sides and reject
                # matches where a steamid appears on both sides in rounds 1-12.
        player_side = (
            first_tick
            .select(["steamid", "side"])
            .unique()
            .group_by("steamid")
            .agg(
                pl.col("side").n_unique().alias("n_sides"),
                pl.col("side").first().alias("first_side"),
            )
                )
        ambiguous = player_side.filter(pl.col("n_sides") != 1)
        if ambiguous.height > 0:
            raise ValueError(f"ambiguous side assignment for {ambiguous.height} players")

        start_round = rounds.get_column("start")
        s_to_append = pl.Series("", [ticks.get_column("tick").min()], dtype=pl.Int64, strict = False)
        start_round.extend(s_to_append)
        starts = (
            ticks
            .filter(pl.col("tick").is_in(start_round.implode()))
            .filter(pl.col("round_num") >= 1)
            .filter(pl.col("side").is_in(["ct", "t"]))
        )
        
        lookup = (
            player_side
            .with_columns(
                pl.when(pl.col("first_side") == "ct")
                .then(pl.lit("t1"))
                .otherwise(pl.lit("t2"))
                .alias("team")
            )
            .select(["steamid", "team"])
        )
        team_counts = lookup.group_by("team").len()
        team_count_d = dict(team_counts.iter_rows())
        if team_count_d.get("t1", 0) == 0 or team_count_d.get("t2", 0) == 0:
            raise ValueError("failed to infer both teams from first-half players")

        t_base = starts.join(lookup, on="steamid", how="inner")
        # Determine t1 side per round from ticks to handle overtime side swaps.
        t1_side = (
            t_base
            .filter(pl.col("team") == "t1")
            .with_columns(pl.col("round_num").cast(pl.Int64))
            .group_by("round_num")
            .agg(pl.col("side").first().alias("t1_side"))
        )


        teams = rounds.with_columns(
            t1_is_ct=(
                pl.when(pl.col("round_num") <= 12).then(True)
                .when(pl.col("round_num") <= 24).then(False)
                .otherwise(
                    pl.when((((pl.col("round_num") - 25) // 6) + 1) % 2 == 0)
                    .then(((pl.col("round_num") - 25) % 6) < 3)
                    .otherwise(((pl.col("round_num") - 25) % 6) >= 3)
                )
            )).with_columns(
            t1_won_round=(
                (pl.col("t1_is_ct") & (pl.col("winner") == "ct")) |
                (~pl.col("t1_is_ct") & (pl.col("winner") == "t"))
            ),
        )

        round_counts = teams.select(
            t1_rounds=pl.col("t1_won_round").sum(),
            t2_rounds=pl.len() - pl.col("t1_won_round").sum(),
        )

        t1_won = round_counts.select("t1_rounds").item() > round_counts.select("t2_rounds").item()


        econ_player = (
            t_base
            .group_by(["team", "round_num", "steamid"])
            .agg(
                pl.col("balance").first().alias("money"),
                pl.col("armor").first().alias("armor"),
            )
        )

        t_inv = t_base.explode("inventory")
        inv = pl.col("inventory").str.to_lowercase()

        econ_inv = (
            t_inv.with_columns(
                t2_rifle=inv.is_in(t2_rifle),
                t1_rifle=inv.is_in(t1_rifle),
                smg=inv.is_in(smg),
                heavy=inv.is_in(heavy),
                t1_pistol=inv.is_in(t1_pistol),
                grenade=inv.is_in(grenade),
                awp=inv.eq("awp"),
                weap_cost=inv.replace_strict(gun_cost, default=0),
            )
            .group_by(["team", "round_num", "steamid"])
            .agg(
                pl.col("t1_rifle").sum().alias("nt1_rifle"),
                pl.col("t2_rifle").sum().alias("nt2_rifle"),
                pl.col("smg").sum().alias("nsmg"),
                pl.col("heavy").sum().alias("nheavy"),
                pl.col("t1_pistol").sum().alias("nt1_pistol"),
                pl.col("grenade").sum().alias("ngrenade"),
                pl.col("awp").sum().alias("nawp"),
                pl.col("weap_cost").sum().alias("inc_weap_cost"),
            )
        )

        econ_features = (
            econ_player.join(econ_inv, on=["team", "round_num", "steamid"])
            .group_by(["team", "round_num"])
            .agg(
                pl.col("money").sum(),
                pl.col("armor").sum(),
                pl.col("nt1_rifle").sum(),
                pl.col("nt2_rifle").sum(),
                pl.col("nsmg").sum(),
                pl.col("nheavy").sum(),
                pl.col("nt1_pistol").sum(),
                pl.col("ngrenade").sum(),
                pl.col("nawp").sum(),
                pl.col("inc_weap_cost").sum(),
            )
        )
        t1 = econ_features.filter(pl.col("team") == "t1").drop("team")
        t2 = econ_features.filter(pl.col("team") == "t2").drop("team")
        econ_features = t1.join(t2, on = "round_num", suffix = "_t2")

        t1_won_round_arr = teams.get_column("t1_won_round").to_numpy()
        feats = teams.with_columns(
            t2_score = (~pl.col("t1_won_round")).cast(pl.Int64).cum_sum().shift(1).fill_null(0),
            t1_score = (pl.col("t1_won_round")).cast(pl.Int64).cum_sum().shift(1).fill_null(0),
            t1_loss_bonus = pl.Series(
                compute_loss_bonus(t1_won_round_arr),
                dtype=pl.Float64,
            ),
            t2_loss_bonus = pl.Series(
                compute_loss_bonus(~t1_won_round_arr),
                dtype=pl.Float64,
            ),
        )
        econ_features = econ_features.with_columns(
            pl.col("round_num")
            .cast(pl.Int64, strict=False)
        )
        feats = feats.select(
            "round_num",
            "t2_score",
            "t1_score",
            "t1_is_ct",
            "t1_loss_bonus",
            "t2_loss_bonus",
        ).join(econ_features, on = "round_num")

        feats = feats.with_columns(
            t1_won = pl.lit(t1_won),
            map = pl.lit(map_played),
            game = pl.lit(game)
        )
        #man fights schema errors for far too long...
        feats = feats.with_columns([
            cs.boolean().cast(pl.Float64),
            cs.integer().cast(pl.Float64),
            cs.float().cast(pl.Float64),
        ])
        dfs.append(feats)
    except Exception as e:
        print(game)
        traceback.print_exc()
        continue
all_feats = pl.concat(dfs)

all_feats.write_csv(f"{DATA_PATH}/features/rbr_wp_feats.csv")

