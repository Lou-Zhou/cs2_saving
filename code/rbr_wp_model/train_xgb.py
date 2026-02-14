import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

import itertools

fix = {
    "anc": "ancient",
    "d": "dust2",
    "inf": "inferno",
    "mi": "mirage",
    "over": "overpass",
    "t": "train",
}

feats = pl.read_csv("/home/lz80/rdf/sp161/shared/cs2_saving/parsed/features/rbr_wp_feats.csv")

def loss_bonus_levels(loss_bonus):
    loss_bonus = pl.col(loss_bonus)
    return(
        pl.when(loss_bonus == 1400).then(pl.lit(0))
        .when(loss_bonus == 1900).then(pl.lit(1))
        .when(loss_bonus == 2400).then(pl.lit(2))
        .when(loss_bonus == 2900).then(pl.lit(3))
        .when(loss_bonus == 3400).then(pl.lit(4))
        .otherwise(pl.lit(-1))
    )

def band_from_cuts(econ: pl.Expr, cuts: list[float]) -> pl.Expr:
    return (
        pl.when(econ < cuts[0]).then(pl.lit(0))
        .when(econ < cuts[1]).then(pl.lit(1))
        .when(econ < cuts[2]).then(pl.lit(2))
        .when(econ < cuts[3]).then(pl.lit(3))
        .when(econ < cuts[4]).then(pl.lit(4))
        .otherwise(pl.lit(5))
    )

def econ_band_expr(econ_col: str, side_col: str) -> pl.Expr:
    econ = pl.col(econ_col)

    ct_band = band_from_cuts(econ, [1600, 2300, 3200, 4200, 5200])
    t_band  = band_from_cuts(econ, [1400, 2000, 2900, 3800, 4800])

    return (
        pl.when(pl.col(side_col) == "ct").then(ct_band)
        .when(pl.col(side_col) == "t").then(t_band)
        .otherwise(pl.lit(None))
    )

features = feats.with_columns(
    round_diff = pl.col("t1_score") - pl.col("t2_score"),
    ot = pl.when(pl.col("round_num") <= 24)
        .then(pl.lit(0))
        .otherwise(((pl.col("round_num") - 25) // 6) + 1)
).with_columns(
    t1_towin = (13 + pl.col("ot") * 3) - pl.col("t1_score"),
    t2_towin = (13 + pl.col("ot") * 3) - pl.col("t2_score"),
    money_diff = pl.col("money") - pl.col("money_t2"),
    armor_diff = pl.col("armor") - pl.col("armor_t2"),
    is_ot = pl.col("ot") != 0,
    inc_weap_diff = pl.col("inc_weap_cost") - pl.col("inc_weap_cost_t2"),
).with_columns(
    round_togo_diff = pl.col("t1_towin") - pl.col("t2_towin"),
    t1_economy = pl.col("money") + pl.col("inc_weap_cost"),
    t2_economy = pl.col("money_t2") + pl.col("inc_weap_cost_t2"),
).with_columns(
    econ_diff = pl.col("t1_economy") - pl.col("t2_economy")
)
features = features.with_columns(
    pl.col("map").cast(pl.Categorical)
).to_dummies(columns=["map"])

features = features.with_columns(
    t1_economy = pl.col("money") + pl.col("inc_weap_cost") + pl.col("armor") / 100 * 650,
    t2_economy = pl.col("money_t2") + pl.col("inc_weap_cost_t2") + pl.col("armor_t2") / 100 * 650,
    t1_lb_band = loss_bonus_levels("t1_loss_bonus"),
    t2_lb_band = loss_bonus_levels("t2_loss_bonus"),
    next_round = pl.col("round_num") + 1
).with_columns(
    lb_band_diff = pl.col("t1_lb_band") - pl.col("t2_lb_band"),
    t1_econ_pp = pl.col("t1_economy") / 5,
    t2_econ_pp = pl.col("t2_economy") / 5,
    t1_side = pl.when(pl.col("t1_is_ct") == 1).then(pl.lit("ct")).otherwise(pl.lit("t")),
    t2_side = pl.when(pl.col("t1_is_ct") == 1).then(pl.lit("t")).otherwise(pl.lit("ct")),
    is_pistol_round = pl.when(pl.col("round_num").is_in([1, 13])).then(pl.lit(1)).otherwise(0)
).with_columns(
    t1_econ_band=econ_band_expr("t1_econ_pp", "t1_side"),
    t2_econ_band=econ_band_expr("t2_econ_pp", "t2_side"),
).with_columns(
    econ_band_diff = pl.col("t1_econ_band") - pl.col("t2_econ_band")
)

feature_cols = ['round_diff', 'econ_diff', 'armor_diff', "round_togo_diff", 'is_pistol_round', 't1_is_ct', 't1_economy', 't2_economy',
    't1_towin', 't2_towin', 'money_diff', 't1_lb_band', 't2_lb_band', 't1_econ_band', 't2_econ_band', 'econ_band_diff', 'lb_band_diff']
feature_cols = feature_cols + [f"map_{m}" for m in ["ancient", "anubis", "dust2", 'inferno', "mirage", "nuke", "overpass", "train"]]
target_col = "t1_won"

X = features.select(feature_cols).to_numpy()
y = features.select(target_col).to_numpy()
groups = features["game"].to_numpy()


param_grid = {
    "max_depth": [4, 6, 8, 10, 12, 14, 16, 18],
    "eta": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1.0, 5.0],
}

base_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "seed": 42,
}

grid_keys = list(param_grid.keys())
grid_values = [param_grid[k] for k in grid_keys]

best_mean = 0
best_params = None
gkf = GroupKFold(n_splits=5)
for values in tqdm(itertools.product(*grid_values)):
    params = base_params.copy()
    params.update(dict(zip(grid_keys, values)))
    fold_logloss = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dvalid = xgb.DMatrix(X[va_idx], label=y[va_idx])

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )

        preds = bst.predict(dvalid)
        auc = roc_auc_score(y[va_idx], preds)
        fold_logloss.append(auc)

    mean_auc = float(np.mean(fold_logloss))
    print(f"params={params} mean_auc={mean_auc:.5f}")
    if mean_auc > best_mean:
        best_mean = mean_auc
        best_params = params

print(f"best mean logloss: {best_mean:.5f}")
print(f"best params: {best_params}")

dtrain = xgb.DMatrix(X, y)

bst = xgb.train(
    best_params,
    dtrain,
    num_boost_round=100000,
    evals=[(dvalid, "valid")],
    verbose_eval=False,
)

bst.save_model("/home/lz80/cs2_saving/stores/model/rbr_xgb.json")