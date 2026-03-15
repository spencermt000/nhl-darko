"""
bootstrap_rapm.py — Game-level bootstrap for per-player RAPM uncertainty.

Resamples games with replacement (200 iterations) and refits Ridge
for each metric to produce empirical standard errors per player.
These replace the diagonal posterior approximation (which is nearly
constant across players) and feed a precision-weighted blend in blend.py.

Output:
  data/rapm_bootstrap_se.csv — player_id, BPR_O_se_boot, BPR_D_se_boot, BPR_se_boot
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV, Ridge

# ── Config ───────────────────────────────────────────────────────────────────
N_BOOTSTRAP      = 200
ALPHA_CANDIDATES = [100, 500, 1000, 5000, 10_000, 50_000]
STRENGTH_FILTER  = "5v5"
DECAY_LAMBDA     = 0.3
EVENTS_PER_60    = 90.0
SEED             = 42

METRICS = {
    "xGF": "xGoal",
    "GF":  "is_goal",
    "SOG": "is_shot_on_goal",
    "TO":  "is_takeaway",
    "GA":  "is_giveaway",
}

BPR_WEIGHTS = {
    "xGF": (0.50, 0.50),
    "SOG": (0.22, 0.22),
    "GF":  (0.15, 0.15),
    "TO":  (0.06, 0.06),
    "GA":  (-0.04, -0.04),
}

SCORE_BINS   = [-99, -2, -1, 0, 1, 2, 99]
SCORE_LABELS = ["down2+", "down1", "tied", "up1", "up2", "up3+"]
SCORE_REF    = "tied"

# ── Goalie exclusion ─────────────────────────────────────────────────────────
print("Identifying goalies...", file=sys.stderr)
goalie_ids = set(
    pd.read_csv("data/raw_pbp.csv", usecols=["event_goalie_id"])
    ["event_goalie_id"].dropna().astype(float).astype(int)
)
print(f"  {len(goalie_ids)} goalie IDs to exclude", file=sys.stderr)

# ── Load PBP ─────────────────────────────────────────────────────────────────
print("Loading clean_pbp.csv...", file=sys.stderr)
df = pd.read_csv("data/clean_pbp.csv")
df = df[
    (df["strength"] == STRENGTH_FILTER) &
    df["home_on_ice"].notna() &
    df["away_on_ice"].notna() &
    df["event_team_type"].notna()
].reset_index(drop=True)

df["score_bucket"] = pd.cut(
    df["score_diff"].fillna(0),
    bins=SCORE_BINS, labels=SCORE_LABELS, right=True,
).astype(str)

max_season = df["season"].max()
df["recency_weight"] = np.exp(-DECAY_LAMBDA * (max_season - df["season"]))

print(f"  {len(df):,} 5v5 events  seasons {df['season'].min()}–{max_season}",
      file=sys.stderr)

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(t.strip()) for t in str(s).split(",") if t.strip()]

def build_player_index(frame):
    ids = set()
    for col in ("home_on_ice", "away_on_ice"):
        for s in frame[col]:
            ids.update(parse_ids(s))
    return sorted(ids - goalie_ids)

def build_design_matrix(frame, player_ids, pid_to_idx):
    n_events  = len(frame)
    n_players = len(player_ids)
    score_cats = [s for s in SCORE_LABELS if s != SCORE_REF]
    n_score    = len(score_cats)
    score_cat_idx = {s: i for i, s in enumerate(score_cats)}
    n_cols = n_players * 2 + n_score

    rows, cols_out, vals = [], [], []
    home_acting = (frame["event_team_type"] == "home").values
    home_ice    = frame["home_on_ice"].values
    away_ice    = frame["away_on_ice"].values
    buckets     = frame["score_bucket"].values

    for i in range(n_events):
        acting_str = home_ice[i] if home_acting[i] else away_ice[i]
        defend_str = away_ice[i] if home_acting[i] else home_ice[i]

        for pid in parse_ids(acting_str):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2);     vals.append(1.0)
        for pid in parse_ids(defend_str):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2 + 1); vals.append(-1.0)

        b = buckets[i]
        if b in score_cat_idx:
            rows.append(i)
            cols_out.append(n_players * 2 + score_cat_idx[b])
            vals.append(1.0)

    X = sparse.csr_matrix(
        (np.array(vals,     dtype=np.float32),
         (np.array(rows,    dtype=np.int32),
          np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )
    return X

# ── Build player index and design matrix (once) ─────────────────────────────
print("Building player index and design matrix...", file=sys.stderr)
player_ids = build_player_index(df)
pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
n_players  = len(player_ids)
X = build_design_matrix(df, player_ids, pid_to_idx)
sample_weight = df["recency_weight"].values

print(f"  {n_players:,} skaters, matrix {X.shape[0]:,} × {X.shape[1]:,}, {X.nnz:,} nnz",
      file=sys.stderr)

# ── Build y arrays for each metric ──────────────────────────────────────────
y_arrays = {}
for metric_name, y_col in METRICS.items():
    y_arrays[metric_name] = df[y_col].fillna(0).astype(float).values

# ── Full-data RidgeCV to select alphas ───────────────────────────────────────
print("\nFull-data RidgeCV to select alphas...", file=sys.stderr)
selected_alphas = {}
for metric_name, y in y_arrays.items():
    model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
    model.fit(X, y, sample_weight=sample_weight)
    selected_alphas[metric_name] = model.alpha_
    print(f"  {metric_name}: alpha={model.alpha_:.0f}", file=sys.stderr)

# ── Precompute game boundaries ───────────────────────────────────────────────
print("\nPrecomputing game boundaries...", file=sys.stderr)
game_ids = df["game_id"].values
unique_games = np.unique(game_ids)
game_starts = np.searchsorted(game_ids, unique_games, side="left")
game_ends   = np.searchsorted(game_ids, unique_games, side="right")
n_games = len(unique_games)
print(f"  {n_games:,} unique games", file=sys.stderr)

# Pre-build index arrays per game for fast concatenation
game_row_indices = [np.arange(game_starts[g], game_ends[g]) for g in range(n_games)]

# ── Bootstrap loop ───────────────────────────────────────────────────────────
print(f"\nBootstrap: {N_BOOTSTRAP} iterations...", file=sys.stderr)
rng = np.random.default_rng(seed=SEED)

# Store per-iteration BPR_O and BPR_D for each player
boot_bpr_o = np.zeros((N_BOOTSTRAP, n_players), dtype=np.float64)
boot_bpr_d = np.zeros((N_BOOTSTRAP, n_players), dtype=np.float64)

for b in range(N_BOOTSTRAP):
    if (b + 1) % 20 == 0 or b == 0:
        print(f"  iteration {b + 1}/{N_BOOTSTRAP}", file=sys.stderr)

    # Resample games with replacement
    sampled_games = rng.choice(n_games, size=n_games, replace=True)
    row_idx = np.concatenate([game_row_indices[g] for g in sampled_games])

    X_b = X[row_idx, :]
    w_b = sample_weight[row_idx]

    # Fit each metric, accumulate BPR per player
    iter_bpr_o = np.zeros(n_players, dtype=np.float64)
    iter_bpr_d = np.zeros(n_players, dtype=np.float64)

    for metric_name, y_full in y_arrays.items():
        y_b = y_full[row_idx]
        alpha = selected_alphas[metric_name]

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_b, y_b, sample_weight=w_b)

        coefs = model.coef_
        w_o, w_d = BPR_WEIGHTS[metric_name]

        for i in range(n_players):
            o_val = coefs[i * 2]     * EVENTS_PER_60
            d_val = coefs[i * 2 + 1] * EVENTS_PER_60
            iter_bpr_o[i] += w_o * o_val
            iter_bpr_d[i] += w_d * d_val

    boot_bpr_o[b] = iter_bpr_o
    boot_bpr_d[b] = iter_bpr_d

# ── Compute bootstrap SEs ────────────────────────────────────────────────────
print("\nComputing bootstrap standard errors...", file=sys.stderr)
se_bpr_o = np.std(boot_bpr_o, axis=0, ddof=1)
se_bpr_d = np.std(boot_bpr_d, axis=0, ddof=1)
se_bpr   = np.sqrt(se_bpr_o ** 2 + se_bpr_d ** 2)

# ── Output ───────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "player_id":      player_ids,
    "BPR_O_se_boot":  np.round(se_bpr_o, 6),
    "BPR_D_se_boot":  np.round(se_bpr_d, 6),
    "BPR_se_boot":    np.round(se_bpr, 6),
})
out.to_csv("data/rapm_bootstrap_se.csv", index=False)
print(f"\nWrote {len(out):,} players to data/rapm_bootstrap_se.csv", file=sys.stderr)

# ── Summary stats ────────────────────────────────────────────────────────────
print("\nBootstrap SE distribution (BPR_se_boot):", file=sys.stderr)
print(out["BPR_se_boot"].describe().round(6).to_string(), file=sys.stderr)

# Show range: lowest SE (most certain) vs highest SE (most uncertain)
sorted_out = out.sort_values("BPR_se_boot")
print(f"\nMost certain (lowest SE):", file=sys.stderr)
print(sorted_out.head(10)[["player_id", "BPR_se_boot"]].to_string(index=False), file=sys.stderr)
print(f"\nLeast certain (highest SE):", file=sys.stderr)
print(sorted_out.tail(10)[["player_id", "BPR_se_boot"]].to_string(index=False), file=sys.stderr)
