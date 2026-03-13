"""
NHL RAPM → BPR-style composite rating.

Design:
  - 5v5 events only, all seasons pooled OR per-season (MODE flag)
  - Goalies excluded from player index
  - Score state (bucketed score_diff) added as covariates to control for score effects
  - Separate O/D columns per skater
  - RidgeCV for regularization (handles small-sample players naturally)
  - BPR composite from weighted RAPM components

Metrics:
  xGF  expected goals for (xGoal, continuous)   — most stable
  GF   goals for (binary)                        — what matters, high variance
  SOG  shots on goal for (binary)                — volume, stabilizes fast
  TO   takeaways (binary, acting = taking team)
  GA   giveaways (binary, acting = giving team; high O = bad)

BPR weights (inspired by Evan Miya / predictive stability):
  Offense: 0.45·xGF_O + 0.20·SOG_O + 0.15·GF_O + 0.12·TO_O − 0.08·GA_O
  Defense: 0.45·xGF_D + 0.20·SOG_D + 0.15·GF_D + 0.12·TO_D − 0.08·GA_D
  BPR_total = BPR_O + BPR_D

Score-state control: [-∞,−3] / −2 / −1 / 0 / +1 / +2 / [+3,+∞]
  Added as 6 dummy columns (tied = reference level) to absorb trailing-team bias.

Output: data/rapm_results.csv   (pooled)
        data/rapm_{season}.csv  (per-season, if MODE = "per_season")
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV

# ── Config ─────────────────────────────────────────────────────────────────────
MODE             = "both"            # "pooled" | "per_season" | "both"
ALPHA_CANDIDATES = [100, 500, 1000, 5000, 10000, 50000]
STRENGTH_FILTER  = "5v5"

METRICS = {              # name → clean_pbp column
    "xGF": "xGoal",
    "GF":  "is_goal",
    "SOG": "is_shot_on_goal",
    "TO":  "is_takeaway",
    "GA":  "is_giveaway",
}

# BPR weights: (O_weight, D_weight)
# GA is negated because high giveaway rate is bad
BPR_WEIGHTS = {
    "xGF": (0.45, 0.45),
    "SOG": (0.20, 0.20),
    "GF":  (0.15, 0.15),
    "TO":  (0.12, 0.12),
    "GA":  (-0.08, -0.08),   # negative → penalize own GAs, reward forcing opponent GAs
}

# 7 edges → 6 intervals → 6 labels
SCORE_BINS   = [-99, -2, -1, 0, 1, 2, 99]
SCORE_LABELS = ["down2+", "down1", "tied", "up1", "up2", "up3+"]
SCORE_REF    = "tied"    # reference level (omitted from dummies)

# ── Load goalie IDs to exclude ──────────────────────────────────────────────────
print("Identifying goalies...", file=sys.stderr)
goalie_ids = set(
    pd.read_csv("data/raw_pbp.csv", usecols=["event_goalie_id"])
    ["event_goalie_id"].dropna().astype(float).astype(int)
)
print(f"  {len(goalie_ids)} goalie IDs to exclude", file=sys.stderr)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading clean_pbp.csv...", file=sys.stderr)
df = pd.read_csv("data/clean_pbp.csv")
df = df[
    (df["strength"] == STRENGTH_FILTER) &
    df["home_on_ice"].notna() &
    df["away_on_ice"].notna() &
    df["event_team_type"].notna()
].reset_index(drop=True)

# Bucket score state
df["score_bucket"] = pd.cut(
    df["score_diff"].fillna(0),
    bins=SCORE_BINS, labels=SCORE_LABELS, right=True
).astype(str)

print(f"  {len(df):,} 5v5 events across seasons {df['season'].min()}–{df['season'].max()}",
      file=sys.stderr)

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(t.strip()) for t in str(s).split(",") if t.strip()]

def build_player_index(frame):
    """Return sorted list of non-goalie player IDs in this frame."""
    ids = set()
    for col in ("home_on_ice", "away_on_ice"):
        for s in frame[col]:
            ids.update(parse_ids(s))
    return sorted(ids - goalie_ids)

def build_design_matrix(frame, player_ids, pid_to_idx):
    """
    Returns sparse matrix X of shape (n_events, n_players*2 + n_score_dummies).
    First n_players*2 cols: player O/D (col = idx*2 for O, idx*2+1 for D)
    Last n_score_dummies cols: score-state dummies (all except reference level)
    """
    n_events   = len(frame)
    n_players  = len(player_ids)   # used for column offsets below
    score_cats = [s for s in SCORE_LABELS if s != SCORE_REF]
    n_score   = len(score_cats)
    score_cat_idx = {s: i for i, s in enumerate(score_cats)}
    n_cols = n_players * 2 + n_score

    rows, cols_out, vals = [], [], []

    home_acting = (frame["event_team_type"] == "home").values
    home_ice    = frame["home_on_ice"].values
    away_ice    = frame["away_on_ice"].values
    buckets     = frame["score_bucket"].values

    for i in range(n_events):
        acting_str  = home_ice[i] if home_acting[i] else away_ice[i]
        defend_str  = away_ice[i] if home_acting[i] else home_ice[i]

        for pid in parse_ids(acting_str):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2);     vals.append(1.0)
        for pid in parse_ids(defend_str):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2 + 1); vals.append(1.0)

        # Score-state dummy
        b = buckets[i]
        if b in score_cat_idx:
            rows.append(i)
            cols_out.append(n_players * 2 + score_cat_idx[b])
            vals.append(1.0)

    X = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )
    return X, n_score

def fit_rapm(frame, X, player_ids, events_per_60):
    """Fit all metrics; return dict {player_id: {metric_O, metric_D, metric_net, ...}}"""
    n_players = len(player_ids)
    results   = {pid: {} for pid in player_ids}

    for metric_name, y_col in METRICS.items():
        y = frame[y_col].fillna(0).astype(float).values
        model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
        model.fit(X, y)
        n_pos = int((y > 0).sum())
        print(f"    {metric_name}: {n_pos:,} events with y>0 | alpha={model.alpha_:.0f}",
              file=sys.stderr)

        coefs = model.coef_   # length = n_players*2 + n_score_dummies
        scale = events_per_60

        for i, pid in enumerate(player_ids):
            o = float(coefs[i * 2])     * scale
            d = float(coefs[i * 2 + 1]) * scale
            results[pid][f"{metric_name}_O"]   = round(o,     4)
            results[pid][f"{metric_name}_D"]   = round(d,     4)
            results[pid][f"{metric_name}_net"] = round(o - d, 4)

    # BPR composite
    for pid in player_ids:
        r = results[pid]
        bpr_o = sum(BPR_WEIGHTS[m][0] * r[f"{m}_O"] for m in METRICS)
        bpr_d = sum(BPR_WEIGHTS[m][1] * r[f"{m}_D"] for m in METRICS)
        r["BPR_O"]   = round(bpr_o, 4)
        r["BPR_D"]   = round(bpr_d, 4)
        r["BPR"]     = round(bpr_o + bpr_d, 4)

    return results

def results_to_df(results, player_names, season=None):
    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "player_id"
    out = out.reset_index()
    out = out.merge(player_names, on="player_id", how="left")
    if season:
        out.insert(1, "season", season)
    return out.sort_values("BPR", ascending=False)

# ── Player name lookup ─────────────────────────────────────────────────────────
pbp_raw = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=["event_player_1_id", "event_player_1_name"],
    dtype=str,
)
pbp_raw = pbp_raw[pbp_raw["event_player_1_id"].str.match(r"^\d+\.?\d*$", na=False)]
pbp_raw["player_id"] = pbp_raw["event_player_1_id"].astype(float).astype(int)
player_names = (
    pbp_raw[["player_id", "event_player_1_name"]]
    .rename(columns={"event_player_1_name": "player_name"})
    .drop_duplicates("player_id")
)

# ── Pooled model ───────────────────────────────────────────────────────────────
def run_pooled(frame, label="pooled"):
    print(f"\n── {label} ({len(frame):,} events) ──────────────", file=sys.stderr)
    player_ids = build_player_index(frame)
    pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    print(f"  {len(player_ids):,} skaters", file=sys.stderr)
    X, _ = build_design_matrix(frame, player_ids, pid_to_idx)
    print(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}, {X.nnz:,} nnz", file=sys.stderr)
    events_per_60 = 90.0   # approximate 5v5 events per team per 60 min
    results = fit_rapm(frame, X, player_ids, events_per_60)
    return results_to_df(results, player_names)

if MODE in ("pooled", "both"):
    out = run_pooled(df)
    out.to_csv("data/rapm_results.csv", index=False)
    print(f"\nTop 15 (pooled) by BPR:", file=sys.stderr)
    print(out.head(15)[["player_name", "BPR_O", "BPR_D", "BPR",
                          "xGF_net", "GF_net", "SOG_net", "TO_net", "GA_net"]]
          .to_string(index=False), file=sys.stderr)

if MODE in ("per_season", "both"):
    season_frames = []
    for season in sorted(df["season"].unique()):
        sf = df[df["season"] == season].reset_index(drop=True)
        if len(sf) < 10000:   # skip very small partial seasons
            print(f"  Skipping season {season} ({len(sf):,} events — too small)", file=sys.stderr)
            continue
        sout = run_pooled(sf, label=f"season {season}")
        sout.insert(1, "season", season)
        season_frames.append(sout)
    if season_frames:
        all_seasons = pd.concat(season_frames, ignore_index=True)
        all_seasons.to_csv("data/rapm_by_season.csv", index=False)
        print(f"\nWrote {len(all_seasons):,} player-season rows to data/rapm_by_season.csv",
              file=sys.stderr)
