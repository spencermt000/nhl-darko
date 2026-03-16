"""
rolling_rapm.py — 3-season rolling window RAPM.

Instead of pooling ALL seasons into one RAPM estimate, fits separate models
on overlapping 3-season windows. This captures player talent at their current
level rather than career averages.

Windows: 2014-2016, 2015-2017, ..., 2022-2024
Each player gets BPR_O, BPR_D, BPR, BPR_se from their most recent window.

Uses fixed alphas from the full-data RidgeCV (data/v2_raw_alphas.json) to
skip cross-validation and run fast (~2-3 min per window).

Inputs:
  data/v2_clean_pbp.csv              Event-level PBP with lineups
  data/v2_raw_alphas.json            Per-metric ridge alphas

Outputs:
  data/v3_rolling_rapm.csv           Per-player per-window RAPM estimates
  data/v3_rolling_rapm_latest.csv    Each player's most recent window estimate
"""

import sys
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

# ── Config ───────────────────────────────────────────────────────────────────
WINDOW_SIZE = 3
STRENGTH_FILTER = "5v5"
EVENTS_PER_60 = 90.0

# BPR weights
W_xGF = 0.50
W_GF  = 0.15
W_SOG = 0.22
W_TO  = 0.06
W_GA  = -0.04

METRICS = {
    "xGF": "xGoal",
    "GF":  "is_goal",
    "SOG": "is_shot_on_goal",
    "TO":  "is_takeaway",
    "GA":  "is_giveaway",
}
BPR_WEIGHTS = {"xGF": W_xGF, "GF": W_GF, "SOG": W_SOG, "TO": W_TO, "GA": W_GA}

# Load alphas
with open("data/v2_raw_alphas.json") as f:
    ALPHAS = json.load(f)


# ── Helper functions ─────────────────────────────────────────────────────────
def parse_ids(s):
    if pd.isna(s):
        return []
    try:
        return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def build_design_matrix(frame, skater_ids, skater_idx, goalie_list, goalie_idx):
    """Build sparse design matrix: skater O/D + goalie columns."""
    n_events = len(frame)
    n_skaters = len(skater_ids)
    n_goalies = len(goalie_list)
    n_cols = n_skaters * 2 + n_goalies

    goalie_offset = n_skaters * 2

    rows, cols_out, vals = [], [], []
    home_acting = (frame["event_team_type"] == "home").values
    home_ice = frame["home_on_ice"].values
    away_ice = frame["away_on_ice"].values

    home_g = frame["home_goalie_id"].values if "home_goalie_id" in frame.columns else None
    away_g = frame["away_goalie_id"].values if "away_goalie_id" in frame.columns else None

    for i in range(n_events):
        acting_str = home_ice[i] if home_acting[i] else away_ice[i]
        defend_str = away_ice[i] if home_acting[i] else home_ice[i]

        for pid in parse_ids(acting_str):
            if pid in skater_idx:
                rows.append(i); cols_out.append(skater_idx[pid] * 2); vals.append(1.0)
        for pid in parse_ids(defend_str):
            if pid in skater_idx:
                rows.append(i); cols_out.append(skater_idx[pid] * 2 + 1); vals.append(-1.0)

        if home_g is not None and not np.isnan(home_g[i]):
            gid = int(home_g[i])
            if gid in goalie_idx:
                rows.append(i); cols_out.append(goalie_offset + goalie_idx[gid]); vals.append(1.0)
        if away_g is not None and not np.isnan(away_g[i]):
            gid = int(away_g[i])
            if gid in goalie_idx:
                rows.append(i); cols_out.append(goalie_offset + goalie_idx[gid]); vals.append(-1.0)

    X = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )
    return X


def fit_window(frame, X, skater_ids):
    """Fit Ridge RAPM on a single window. Returns dict of player results."""
    n_skaters = len(skater_ids)
    results = {pid: {} for pid in skater_ids}

    for metric_name, y_col in METRICS.items():
        y = frame[y_col].fillna(0).astype(float).values
        alpha = ALPHAS[metric_name]

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y)
        coefs = model.coef_

        scale = EVENTS_PER_60

        for si, pid in enumerate(skater_ids):
            o_coef = float(coefs[si * 2]) * scale
            d_coef = float(coefs[si * 2 + 1]) * scale
            results[pid][f"{metric_name}_O"] = round(o_coef, 4)
            results[pid][f"{metric_name}_D"] = round(d_coef, 4)
            results[pid][f"{metric_name}_net"] = round(o_coef + d_coef, 4)

        # Compute SEs via hat matrix diagonal approximation
        # SE ≈ residual_std / sqrt(diagonal of X'X + αI)
        residuals = y - model.predict(X)
        res_std = float(np.std(residuals))

        # Approximate SE: use the mean diagonal of (X'X + αI)^-1
        # For sparse Ridge, exact SE is expensive. Use a simpler approximation:
        # SE ≈ res_std / sqrt(event_count_for_player * something)
        # Better: use the column norms of X
        col_norms_sq = np.array(X.power(2).sum(axis=0)).flatten()

        for si, pid in enumerate(skater_ids):
            o_norm = col_norms_sq[si * 2]
            d_norm = col_norms_sq[si * 2 + 1]
            o_se = res_std * scale / np.sqrt(o_norm + alpha) if o_norm > 0 else 1.0
            d_se = res_std * scale / np.sqrt(d_norm + alpha) if d_norm > 0 else 1.0
            results[pid][f"{metric_name}_O_se"] = round(o_se, 4)
            results[pid][f"{metric_name}_D_se"] = round(d_se, 4)

    # Compute BPR from per-metric coefficients
    for pid in skater_ids:
        r = results[pid]
        bpr_o = sum(r.get(f"{m}_O", 0) * BPR_WEIGHTS[m] for m in METRICS)
        bpr_d = sum(r.get(f"{m}_D", 0) * BPR_WEIGHTS[m] for m in METRICS)
        r["BPR_O"] = round(bpr_o, 4)
        r["BPR_D"] = round(bpr_d, 4)
        r["BPR"] = round(bpr_o + bpr_d, 4)

        # BPR SE via quadrature
        bpr_o_se = np.sqrt(sum(
            (r.get(f"{m}_O_se", 0) * abs(BPR_WEIGHTS[m])) ** 2 for m in METRICS
        ))
        bpr_d_se = np.sqrt(sum(
            (r.get(f"{m}_D_se", 0) * abs(BPR_WEIGHTS[m])) ** 2 for m in METRICS
        ))
        r["BPR_O_se"] = round(bpr_o_se, 4)
        r["BPR_D_se"] = round(bpr_d_se, 4)
        r["BPR_se"] = round(np.sqrt(bpr_o_se ** 2 + bpr_d_se ** 2), 4)

    return results


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading v2_clean_pbp.csv...", file=sys.stderr)
df = pd.read_csv("data/v2_clean_pbp.csv")
df = df[df["strength_state"] == STRENGTH_FILTER].copy()
print(f"  {len(df):,} 5v5 events", file=sys.stderr)

# Player name/position lookup
player_info = pd.read_csv("data/v2_final_ratings.csv", usecols=["player_id", "player_name", "position"])
player_info["player_id"] = player_info["player_id"].astype(int)
name_lookup = player_info.set_index("player_id")["player_name"].to_dict()
pos_lookup = player_info.set_index("player_id")["position"].to_dict()

seasons = sorted(df["season"].unique())
print(f"  Seasons: {seasons}", file=sys.stderr)


# ── Fit rolling windows ─────────────────────────────────────────────────────
all_results = []

for start_idx in range(len(seasons) - WINDOW_SIZE + 1):
    window_seasons = seasons[start_idx:start_idx + WINDOW_SIZE]
    window_label = f"{window_seasons[0]}-{window_seasons[-1]}"

    print(f"\n── Window {window_label} ──", file=sys.stderr)

    wf = df[df["season"].isin(window_seasons)].copy()
    print(f"  {len(wf):,} events", file=sys.stderr)

    # Build player and goalie indices
    skater_pids = set()
    for col in ("home_on_ice", "away_on_ice"):
        for s in wf[col].dropna():
            skater_pids.update(parse_ids(s))
    skater_ids = sorted(skater_pids)
    skater_idx = {pid: i for i, pid in enumerate(skater_ids)}

    goalie_ids = set()
    for col in ("home_goalie_id", "away_goalie_id"):
        goalie_ids.update(wf[col].dropna().astype(int).unique())
    goalie_list = sorted(goalie_ids)
    goalie_idx = {gid: i for i, gid in enumerate(goalie_list)}

    print(f"  {len(skater_ids):,} skaters, {len(goalie_list)} goalies", file=sys.stderr)

    # Build design matrix
    X = build_design_matrix(wf, skater_ids, skater_idx, goalie_list, goalie_idx)
    print(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}", file=sys.stderr)

    # Fit
    results = fit_window(wf, X, skater_ids)

    # Collect results
    for pid, r in results.items():
        r["player_id"] = pid
        r["window"] = window_label
        r["window_start"] = window_seasons[0]
        r["window_end"] = window_seasons[-1]
        r["player_name"] = name_lookup.get(pid, "?")
        r["position"] = pos_lookup.get(pid, "?")
        all_results.append(r)

    # Quick sanity check
    top5 = sorted(results.items(), key=lambda x: x[1]["BPR"], reverse=True)[:5]
    print(f"  Top 5 BPR:", file=sys.stderr)
    for pid, r in top5:
        pname = name_lookup.get(pid, str(pid))
        print(f"    {pname:25s} BPR_O={r['BPR_O']:+.3f} BPR_D={r['BPR_D']:+.3f} BPR={r['BPR']:+.3f}", file=sys.stderr)


# ── Save all window results ─────────────────────────────────────────────────
print(f"\n── Saving results ──", file=sys.stderr)
rolling_df = pd.DataFrame(all_results)

# Select key columns
key_cols = [
    "player_id", "player_name", "position", "window", "window_start", "window_end",
    "BPR_O", "BPR_D", "BPR", "BPR_O_se", "BPR_D_se", "BPR_se",
]
# Add per-metric columns
for m in METRICS:
    key_cols.extend([f"{m}_O", f"{m}_D", f"{m}_net"])

key_cols = [c for c in key_cols if c in rolling_df.columns]
rolling_df = rolling_df[key_cols]
rolling_df.to_csv("data/v3_rolling_rapm.csv", index=False)
print(f"  {len(rolling_df):,} player-window rows → data/v3_rolling_rapm.csv", file=sys.stderr)

# Latest window per player: the most recent window they appear in
latest = (
    rolling_df.sort_values("window_end", ascending=False)
    .groupby("player_id")
    .first()
    .reset_index()
)
latest.to_csv("data/v3_rolling_rapm_latest.csv", index=False)
print(f"  {len(latest):,} players (latest window) → data/v3_rolling_rapm_latest.csv", file=sys.stderr)

# Check key players
print("\nKey player BPR (latest window):", file=sys.stderr)
for name in ["Connor.McDavid", "Nikita.Kucherov", "Auston.Matthews", "Kirill.Kaprizov",
             "Nathan.MacKinnon", "Cale.Makar", "Charlie.McAvoy", "Leon.Draisaitl"]:
    row = latest[latest["player_name"] == name]
    if len(row):
        r = row.iloc[0]
        print(f"  {name:25s} [{r['window']}] BPR_O={r['BPR_O']:+.4f} BPR_D={r['BPR_D']:+.4f} BPR={r['BPR']:+.4f}", file=sys.stderr)

print("\nDone.", file=sys.stderr)
