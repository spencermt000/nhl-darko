"""
NHL RAPM v2 — BPR-style composite with Bayesian enhancements.

New in v2 (vs v1):
  1. Exponential recency weighting in pooled model  (recent seasons > old)
  2. Position lookup (C/L/R → F, D → D) from MoneyPuck shots file
  3. Career-average prior for per-season fits  (pooled coefs = ridge target)
  4. Position-group fallback prior for new players  (shrink toward F/D mean)
  5. Diagonal posterior uncertainty  (se columns for every metric + BPR)

Two-pass RAPM (teammate/opponent quality adjustment):
  Per-season fits augment the design matrix with 2 scalar covariates:
    acting_quality   = mean pooled BPR_O of the 5 acting-team players
    defending_quality = mean pooled BPR_D of the 5 defending-team players
  This explicitly controls for context strength, so players carrying weak lines
  or facing elite opponents are adjusted accordingly.

Output:
  data/rapm_results.csv       pooled ratings (BPR, BPR_se, component se's)
  data/rapm_by_season.csv     per-season ratings (career-avg prior + quality covars)
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV

# ── Config ───────────────────────────────────────────────────────────────────
MODE             = "both"           # "pooled" | "per_season" | "both"
ALPHA_CANDIDATES = [100, 500, 1000, 5000, 10_000, 50_000]
STRENGTH_FILTER  = "5v5"
DECAY_LAMBDA     = 0.3              # pooled recency: exp(-0.3 * years_ago)
                                    # 1yr ago → 0.74x, 3yr → 0.41x, 6yr → 0.17x

METRICS = {
    "xGF": "xGoal",
    "GF":  "is_goal",
    "SOG": "is_shot_on_goal",
    "TO":  "is_takeaway",
    "GA":  "is_giveaway",
}

BPR_WEIGHTS = {                     # (O_weight, D_weight)
    "xGF": ( 0.50,  0.50),          # bumped up: most stable predictor
    "SOG": ( 0.22,  0.22),
    "GF":  ( 0.15,  0.15),
    "TO":  ( 0.06,  0.06),          # halved: noisy stat
    "GA":  (-0.04, -0.04),          # halved: noisy stat
}

SCORE_BINS   = [-99, -2, -1, 0, 1, 2, 99]
SCORE_LABELS = ["down2+", "down1", "tied", "up1", "up2", "up3+"]
SCORE_REF    = "tied"               # omitted from dummies (reference level)

EVENTS_PER_60 = 90.0               # approx 5v5 events per team per 60 min

# ── Goalie exclusion ─────────────────────────────────────────────────────────
print("Identifying goalies...", file=sys.stderr)
goalie_ids = set(
    pd.read_csv("data/raw_pbp.csv", usecols=["event_goalie_id"])
    ["event_goalie_id"].dropna().astype(float).astype(int)
)
print(f"  {len(goalie_ids)} goalie IDs to exclude", file=sys.stderr)

# ── Position lookup from MoneyPuck ───────────────────────────────────────────
print("Building position lookup...", file=sys.stderr)
_mp_pos = pd.read_csv(
    "data/shots_2007-2024.csv",
    usecols=["shooterPlayerId", "playerPositionThatDidEvent"],
    dtype={"shooterPlayerId": "Int64"},
).dropna(subset=["shooterPlayerId"])
_mp_pos["shooterPlayerId"] = _mp_pos["shooterPlayerId"].astype(int)
_mp_pos["pos"] = _mp_pos["playerPositionThatDidEvent"].map(
    lambda p: "F" if p in ("C", "L", "R") else ("D" if p == "D" else None)
)
_mp_pos = _mp_pos[_mp_pos["pos"].notna()]
pid_to_pos: dict[int, str] = (
    _mp_pos.groupby("shooterPlayerId")["pos"]
    .agg(lambda s: s.mode().iloc[0])   # most common position for player
    .to_dict()
)
print(f"  {len(pid_to_pos):,} skaters with position data", file=sys.stderr)
del _mp_pos

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

# ── Player name lookup ────────────────────────────────────────────────────────
_raw = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=["event_player_1_id", "event_player_1_name"],
    dtype=str,
)
_raw = _raw[_raw["event_player_1_id"].str.match(r"^\d+\.?\d*$", na=False)]
_raw["player_id"] = _raw["event_player_1_id"].astype(float).astype(int)
player_names = (
    _raw[["player_id", "event_player_1_name"]]
    .rename(columns={"event_player_1_name": "player_name"})
    .drop_duplicates("player_id")
)
del _raw

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(t.strip()) for t in str(s).split(",") if t.strip()]

def build_player_index(frame) -> list[int]:
    ids = set()
    for col in ("home_on_ice", "away_on_ice"):
        for s in frame[col]:
            ids.update(parse_ids(s))
    return sorted(ids - goalie_ids)

def build_design_matrix(frame, player_ids, pid_to_idx, pass1_quality=None):
    """
    Returns sparse CSR matrix of shape (n_events, n_players*2 + n_score_dummies [+ 2]).
    Column layout: [p0_O, p0_D, p1_O, p1_D, ..., score_0, ..., [acting_q, defending_q]]

    Acting-team players: O col = +1.  Defending-team players: D col = +1.

    pass1_quality : dict {player_id: (BPR_O, BPR_D)} from pooled pass-1 ratings.
        If provided, 2 extra columns are appended:
          col[-2] = mean BPR_O of acting team  (teammate context)
          col[-1] = mean BPR_D of defending team  (opponent context)
        Player extraction ignores these trailing columns (uses i*2 / i*2+1 offsets).
    """
    n_events  = len(frame)
    n_players = len(player_ids)      # used for col offsets below
    score_cats = [s for s in SCORE_LABELS if s != SCORE_REF]
    n_score    = len(score_cats)
    score_cat_idx = {s: i for i, s in enumerate(score_cats)}
    n_cols = n_players * 2 + n_score

    rows, cols_out, vals = [], [], []
    home_acting = (frame["event_team_type"] == "home").values
    home_ice    = frame["home_on_ice"].values
    away_ice    = frame["away_on_ice"].values
    buckets     = frame["score_bucket"].values

    # Optional quality covariate arrays (filled below if pass1_quality provided)
    acting_q   = np.zeros(n_events, dtype=np.float32) if pass1_quality else None
    defending_q = np.zeros(n_events, dtype=np.float32) if pass1_quality else None

    for i in range(n_events):
        acting_str = home_ice[i] if home_acting[i] else away_ice[i]
        defend_str = away_ice[i] if home_acting[i] else home_ice[i]

        acting_pids  = parse_ids(acting_str)
        defending_pids = parse_ids(defend_str)

        for pid in acting_pids:
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2);     vals.append(1.0)
        for pid in defending_pids:
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2 + 1); vals.append(-1.0)

        b = buckets[i]
        if b in score_cat_idx:
            rows.append(i)
            cols_out.append(n_players * 2 + score_cat_idx[b])
            vals.append(1.0)

        if pass1_quality is not None:
            a_vals = [pass1_quality[p][0] for p in acting_pids   if p in pass1_quality]
            d_vals = [pass1_quality[p][1] for p in defending_pids if p in pass1_quality]
            acting_q[i]    = float(np.mean(a_vals)) if a_vals else 0.0
            defending_q[i] = float(np.mean(d_vals)) if d_vals else 0.0

    X = sparse.csr_matrix(
        (np.array(vals,     dtype=np.float32),
         (np.array(rows,    dtype=np.int32),
          np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )

    if pass1_quality is not None:
        q_mat = sparse.csr_matrix(
            np.column_stack([acting_q, defending_q]).astype(np.float32)
        )
        X = sparse.hstack([X, q_mat], format="csr")

    return X

def compute_se(X, y, coefs, alpha, sample_weight=None):
    """
    Diagonal approximation of posterior standard errors.

    For ridge:  Var(coef_j) ≈ sigma² / (n_j + alpha)
    where n_j = effective event count for column j.

    With sample weights: n_j = sum_i(w_i * x_ij).
    Since X is binary this equals sum of weights for events involving that column.
    """
    residuals = y - X.dot(coefs)
    if sample_weight is not None:
        w = sample_weight
        residual_var = np.dot(w, residuals ** 2) / np.sum(w)
        # effective weighted count per column
        col_effective = np.array(X.T.dot(w)).flatten()
    else:
        residual_var = np.mean(residuals ** 2)
        col_effective = np.array(X.sum(axis=0)).flatten()

    residual_std = np.sqrt(max(residual_var, 1e-12))
    se = residual_std / np.sqrt(np.maximum(col_effective + alpha, 1.0))
    return se

def fit_rapm(frame, X, player_ids, prior_coefs=None, sample_weight=None):
    """
    Fit all metrics; return (results_dict, all_coefs_dict).

    prior_coefs : dict {metric_name: np.ndarray of shape (n_cols,)}
        Ridge is shifted so coefs shrink toward prior_coefs instead of zero.
        Implementation: y_adj = y - X @ mu, fit ridge on y_adj, add mu back.

    sample_weight : np.ndarray of shape (n_events,), optional.
    """
    results   = {pid: {} for pid in player_ids}
    all_coefs = {}

    for metric_name, y_col in METRICS.items():
        y  = frame[y_col].fillna(0).astype(float).values
        mu = prior_coefs.get(metric_name) if prior_coefs else None

        # Pad prior with zeros for any extra columns (e.g. quality covariates)
        if mu is not None and len(mu) < X.shape[1]:
            mu = np.concatenate([mu, np.zeros(X.shape[1] - len(mu))])
        y_fit = y - X.dot(mu) if mu is not None else y

        model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
        model.fit(X, y_fit, **({"sample_weight": sample_weight} if sample_weight is not None else {}))

        alpha = model.alpha_
        coefs = model.coef_ + (mu if mu is not None else 0)
        all_coefs[metric_name] = coefs

        se = compute_se(X, y, coefs, alpha, sample_weight)

        n_pos = int((y > 0).sum())
        print(f"    {metric_name}: {n_pos:,} events with y>0 | alpha={alpha:.0f}",
              file=sys.stderr)

        scale = EVENTS_PER_60
        for i, pid in enumerate(player_ids):
            o    = float(coefs[i * 2])     * scale
            d    = float(coefs[i * 2 + 1]) * scale
            o_se = float(se[i * 2])        * scale
            d_se = float(se[i * 2 + 1])    * scale
            results[pid][f"{metric_name}_O"]    = round(o,     4)
            results[pid][f"{metric_name}_D"]    = round(d,     4)
            results[pid][f"{metric_name}_net"]  = round(o - d, 4)
            results[pid][f"{metric_name}_O_se"] = round(o_se,  4)
            results[pid][f"{metric_name}_D_se"] = round(d_se,  4)

    # BPR composite + propagated uncertainty (quadrature; metrics are independent regressions)
    for pid in player_ids:
        r = results[pid]
        bpr_o    = sum(BPR_WEIGHTS[m][0] * r[f"{m}_O"] for m in METRICS)
        bpr_d    = sum(BPR_WEIGHTS[m][1] * r[f"{m}_D"] for m in METRICS)
        bpr_o_se = np.sqrt(sum((BPR_WEIGHTS[m][0] * r[f"{m}_O_se"]) ** 2 for m in METRICS))
        bpr_d_se = np.sqrt(sum((BPR_WEIGHTS[m][1] * r[f"{m}_D_se"]) ** 2 for m in METRICS))
        r["BPR_O"]    = round(bpr_o, 4)
        r["BPR_D"]    = round(bpr_d, 4)
        r["BPR"]      = round(bpr_o + bpr_d, 4)
        r["BPR_O_se"] = round(bpr_o_se, 4)
        r["BPR_D_se"] = round(bpr_d_se, 4)
        r["BPR_se"]   = round(np.sqrt(bpr_o_se ** 2 + bpr_d_se ** 2), 4)

    return results, all_coefs

def results_to_df(results, season=None):
    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "player_id"
    out = out.reset_index()
    out = out.merge(player_names, on="player_id", how="left")
    out["position"] = out["player_id"].map(pid_to_pos).fillna("?")
    if season is not None:
        out.insert(1, "season", season)
    return out.sort_values("BPR", ascending=False)

def run_frame(frame, label, prior_coefs=None, sample_weight=None, pass1_quality=None):
    """Build index + design matrix, fit, return (df, player_ids, pid_to_idx, all_coefs)."""
    print(f"\n── {label} ({len(frame):,} events) ──────────────", file=sys.stderr)
    player_ids = build_player_index(frame)
    pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
    print(f"  {len(player_ids):,} skaters", file=sys.stderr)
    X = build_design_matrix(frame, player_ids, pid_to_idx, pass1_quality=pass1_quality)
    extra = " + 2 quality covars" if pass1_quality else ""
    print(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}, {X.nnz:,} nnz{extra}", file=sys.stderr)
    results, all_coefs = fit_rapm(frame, X, player_ids,
                                   prior_coefs=prior_coefs, sample_weight=sample_weight)
    return results_to_df(results), player_ids, pid_to_idx, all_coefs

def build_pos_group_means(player_ids, pid_to_idx, coefs_by_metric):
    """
    Compute per-position (F/D) mean coef for O and D, per metric.
    Used as fallback prior for new players in per-season fits.
    Returns: dict {metric: {pos: {"O": float, "D": float}}}
    """
    pos_sums  = {m: {"F": {"O": [], "D": []}, "D": {"O": [], "D": []}} for m in METRICS}
    for pid in player_ids:
        pos = pid_to_pos.get(pid, "F")
        if pos not in ("F", "D"):
            continue
        i = pid_to_idx[pid]
        for m, arr in coefs_by_metric.items():
            pos_sums[m][pos]["O"].append(arr[i * 2])
            pos_sums[m][pos]["D"].append(arr[i * 2 + 1])

    means = {}
    for m in METRICS:
        means[m] = {}
        for pos in ("F", "D"):
            means[m][pos] = {
                "O": float(np.mean(pos_sums[m][pos]["O"])) if pos_sums[m][pos]["O"] else 0.0,
                "D": float(np.mean(pos_sums[m][pos]["D"])) if pos_sums[m][pos]["D"] else 0.0,
            }
    return means

def build_season_prior(season_player_ids, pooled_pid_to_idx, pooled_coefs, pos_means):
    """
    Map pooled coefs into the per-season player index as ridge prior.
    Players absent from pooled → position-group mean fallback.
    Score-dummy columns get zero prior (no score-state carry-over).
    """
    score_cats = [s for s in SCORE_LABELS if s != SCORE_REF]
    n_cols_s   = len(season_player_ids) * 2 + len(score_cats)
    prior = {}
    for m, arr in pooled_coefs.items():
        mu = np.zeros(n_cols_s, dtype=np.float64)
        for i, pid in enumerate(season_player_ids):
            if pid in pooled_pid_to_idx:
                pi        = pooled_pid_to_idx[pid]
                mu[i * 2]     = arr[pi * 2]
                mu[i * 2 + 1] = arr[pi * 2 + 1]
            else:
                pos           = pid_to_pos.get(pid, "F")
                pg            = pos_means[m].get(pos, {"O": 0.0, "D": 0.0})
                mu[i * 2]     = pg["O"]
                mu[i * 2 + 1] = pg["D"]
        prior[m] = mu
    return prior

# ── Pooled model ──────────────────────────────────────────────────────────────
pooled_ids        = None
pooled_pid_to_idx = None
pooled_coefs      = None
pos_means         = None
pass1_quality     = None

if MODE in ("pooled", "both"):
    sample_weight = df["recency_weight"].values
    out, pooled_ids, pooled_pid_to_idx, pooled_coefs = run_frame(
        df, label="pooled", sample_weight=sample_weight
    )
    out.to_csv("data/rapm_results.csv", index=False)
    print(f"\nTop 15 (pooled) by BPR:", file=sys.stderr)
    cols = ["player_name", "position", "BPR_O", "BPR_D", "BPR", "BPR_se",
            "xGF_net", "GF_net", "SOG_net", "TO_net", "GA_net"]
    print(out.head(15)[cols].to_string(index=False), file=sys.stderr)

    # Precompute position group means for per-season fallback prior
    pos_means = build_pos_group_means(pooled_ids, pooled_pid_to_idx, pooled_coefs)

    # Build pass-1 quality lookup for two-pass per-season fits
    # Keys: player_id  Values: (BPR_O, BPR_D) in per-60 units
    pass1_quality = {
        int(row["player_id"]): (float(row["BPR_O"]), float(row["BPR_D"]))
        for _, row in out.iterrows()
    }
    print(f"  Pass-1 quality lookup: {len(pass1_quality):,} players", file=sys.stderr)

# ── Per-season models ─────────────────────────────────────────────────────────
if MODE in ("per_season", "both"):
    if pooled_coefs is None:
        # Standalone per-season run (no pooled prior available)
        print("Warning: per_season mode without pooled — priors will be zero.", file=sys.stderr)

    season_frames = []
    for season in sorted(df["season"].unique()):
        sf = df[df["season"] == season].reset_index(drop=True)
        if len(sf) < 10_000:
            print(f"  Skipping season {season} ({len(sf):,} events — too small)",
                  file=sys.stderr)
            continue

        # Build season-specific prior from pooled coefs
        if pooled_coefs is not None:
            season_pids   = build_player_index(sf)
            season_prior  = build_season_prior(
                season_pids, pooled_pid_to_idx, pooled_coefs, pos_means
            )
        else:
            season_pids  = None
            season_prior = None

        sout, _, _, _ = run_frame(
            sf, label=f"season {season}",
            prior_coefs=season_prior,
            pass1_quality=pass1_quality,
        )
        sout.insert(1, "season", season)
        season_frames.append(sout)

    if season_frames:
        all_seasons = pd.concat(season_frames, ignore_index=True)
        all_seasons.to_csv("data/rapm_by_season.csv", index=False)
        print(f"\nWrote {len(all_seasons):,} player-season rows to data/rapm_by_season.csv",
              file=sys.stderr)
