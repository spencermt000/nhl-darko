"""
RAPM v2 — Prior-informed Bayesian RAPM with goalie controls.

Two modes:
  --mode=raw    Standard RidgeCV (no prior). Used as training target for box_prior.py.
  --mode=prior  Data-driven alpha from R² calibration, box score prior mean.

Enhancements over v1:
  - Goalie columns in design matrix (separate from skaters)
  - Zone start dummies (OZ, DZ; NZ=reference)
  - Period dummies (P2, P3; P1=reference)
  - Rest differential (continuous control)
  - Prior-informed ridge (Bayesian posterior via shifted target)

Output:
  data/v2_rapm_raw.csv          (--mode=raw) Uninformed RAPM
  data/v2_rapm_results.csv      (--mode=prior) Prior-informed pooled ratings
  data/v2_rapm_by_season.csv    (--mode=prior) Prior-informed per-season ratings
"""

import sys
import json
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge

# ── Parse mode from command line ─────────────────────────────────────────────
MODE = "prior"  # default
for arg in sys.argv[1:]:
    if arg.startswith("--mode="):
        MODE = arg.split("=")[1]
assert MODE in ("raw", "prior"), f"Unknown mode: {MODE}"
print(f"Mode: {MODE}", file=sys.stderr)

# ── Config ───────────────────────────────────────────────────────────────────
ALPHA_CANDIDATES = [100, 500, 1000, 5000, 10_000, 50_000]
STRENGTH_FILTER  = "5v5"  # matches strength_state column
DECAY_LAMBDA     = 0.3
EVENTS_PER_60    = 90.0

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

# ── Load PBP ─────────────────────────────────────────────────────────────────
print("Loading v2_clean_pbp.csv...", file=sys.stderr)
df = pd.read_csv("data/v2_clean_pbp.csv")
df = df[
    (df["strength_state"] == STRENGTH_FILTER) &
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

# Standardize rest differential
rest = df["rest_differential"].fillna(0).values
rest_mean = rest.mean()
rest_std = max(rest.std(), 1e-6)
df["rest_diff_std"] = (rest - rest_mean) / rest_std

print(f"  {len(df):,} 5v5 events  seasons {df['season'].min()}–{max_season}", file=sys.stderr)

# ── Player name lookup ───────────────────────────────────────────────────────
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

# ── Position lookup ──────────────────────────────────────────────────────────
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
pid_to_pos = (
    _mp_pos.groupby("shooterPlayerId")["pos"]
    .agg(lambda s: s.mode().iloc[0])
    .to_dict()
)
del _mp_pos

# ── Goalie IDs ───────────────────────────────────────────────────────────────
# Build set of all goalie IDs that appear in the data
goalie_ids = set()
for col in ("home_goalie_id", "away_goalie_id"):
    goalie_ids.update(df[col].dropna().astype(int).unique())
print(f"  {len(goalie_ids)} unique goalies in 5v5 data", file=sys.stderr)

# Goalie name lookup from raw_pbp
_raw_g = pd.read_csv(
    "data/raw_pbp.csv",
    usecols=["event_goalie_name", "event_goalie_id"],
)
_raw_g = _raw_g.dropna().drop_duplicates()
_raw_g["event_goalie_id"] = _raw_g["event_goalie_id"].astype(int)
goalie_names = dict(zip(_raw_g["event_goalie_id"], _raw_g["event_goalie_name"]))
del _raw_g

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(t.strip()) for t in str(s).split(",") if t.strip()]

def build_player_index(frame):
    """Build sorted list of all skater IDs (excluding goalies from lineup columns)."""
    ids = set()
    for col in ("home_on_ice", "away_on_ice"):
        for s in frame[col]:
            ids.update(parse_ids(s))
    return sorted(ids - goalie_ids)

def build_goalie_index(frame):
    """Build sorted list of all goalie IDs from goalie columns."""
    gids = set()
    for col in ("home_goalie_id", "away_goalie_id"):
        gids.update(frame[col].dropna().astype(int).unique())
    return sorted(gids)

def build_design_matrix(frame, skater_ids, skater_idx, goalie_list, goalie_idx,
                         pass1_quality=None):
    """
    Sparse CSR design matrix.

    Column layout:
      [s0_O, s0_D, s1_O, s1_D, ...,           # skater O/D (2 * n_skaters)
       g0, g1, ...,                              # goalie columns (n_goalies)
       score_0, ..., score_4,                    # score state dummies (5)
       zone_OZ, zone_DZ,                         # zone start dummies (2)
       period_2, period_3,                        # period dummies (2)
       rest_diff,                                 # rest differential (1)
       [acting_q, defending_q]]                   # optional quality covars (2)
    """
    n_events   = len(frame)
    n_skaters  = len(skater_ids)
    n_goalies  = len(goalie_list)
    score_cats = [s for s in SCORE_LABELS if s != SCORE_REF]
    n_score    = len(score_cats)
    score_cat_idx = {s: i for i, s in enumerate(score_cats)}

    # Column offsets
    goalie_offset = n_skaters * 2
    score_offset  = goalie_offset + n_goalies
    zone_offset   = score_offset + n_score
    period_offset = zone_offset + 2
    rest_offset   = period_offset + 2
    n_cols        = rest_offset + 1

    rows, cols_out, vals = [], [], []
    home_acting = (frame["event_team_type"] == "home").values
    home_ice    = frame["home_on_ice"].values
    away_ice    = frame["away_on_ice"].values
    buckets     = frame["score_bucket"].values
    zones       = frame["zone_start"].values if "zone_start" in frame.columns else None
    periods     = frame["period"].values if "period" in frame.columns else None
    rest_vals   = frame["rest_diff_std"].values if "rest_diff_std" in frame.columns else None

    # Goalie IDs per event
    home_g = frame["home_goalie_id"].values if "home_goalie_id" in frame.columns else None
    away_g = frame["away_goalie_id"].values if "away_goalie_id" in frame.columns else None

    # Quality covariates
    acting_q   = np.zeros(n_events, dtype=np.float32) if pass1_quality else None
    defending_q = np.zeros(n_events, dtype=np.float32) if pass1_quality else None

    for i in range(n_events):
        acting_str = home_ice[i] if home_acting[i] else away_ice[i]
        defend_str = away_ice[i] if home_acting[i] else home_ice[i]

        acting_pids  = parse_ids(acting_str)
        defending_pids = parse_ids(defend_str)

        # Skater columns: acting team O = +1, defending team D = -1
        for pid in acting_pids:
            if pid in skater_idx:
                rows.append(i); cols_out.append(skater_idx[pid] * 2);     vals.append(1.0)
        for pid in defending_pids:
            if pid in skater_idx:
                rows.append(i); cols_out.append(skater_idx[pid] * 2 + 1); vals.append(-1.0)

        # Goalie columns: home goalie +1, away goalie -1
        if home_g is not None and not np.isnan(home_g[i]):
            gid = int(home_g[i])
            if gid in goalie_idx:
                rows.append(i); cols_out.append(goalie_offset + goalie_idx[gid]); vals.append(1.0)
        if away_g is not None and not np.isnan(away_g[i]):
            gid = int(away_g[i])
            if gid in goalie_idx:
                rows.append(i); cols_out.append(goalie_offset + goalie_idx[gid]); vals.append(-1.0)

        # Score state
        b = buckets[i]
        if b in score_cat_idx:
            rows.append(i); cols_out.append(score_offset + score_cat_idx[b]); vals.append(1.0)

        # Zone start dummies (OZ=0, DZ=1; NZ=reference)
        if zones is not None:
            z = zones[i]
            if z == "OZ":
                rows.append(i); cols_out.append(zone_offset); vals.append(1.0)
            elif z == "DZ":
                rows.append(i); cols_out.append(zone_offset + 1); vals.append(1.0)

        # Period dummies (P2=0, P3=1; P1=reference)
        if periods is not None:
            p = periods[i]
            if p == 2:
                rows.append(i); cols_out.append(period_offset); vals.append(1.0)
            elif p >= 3:
                rows.append(i); cols_out.append(period_offset + 1); vals.append(1.0)

        # Rest differential (continuous)
        if rest_vals is not None:
            rv = rest_vals[i]
            if rv != 0 and not np.isnan(rv):
                rows.append(i); cols_out.append(rest_offset); vals.append(float(rv))

        # Quality covariates
        if pass1_quality is not None:
            a_vals = [pass1_quality[p][0] for p in acting_pids   if p in pass1_quality]
            d_vals = [pass1_quality[p][1] for p in defending_pids if p in pass1_quality]
            acting_q[i]    = float(np.mean(a_vals)) if a_vals else 0.0
            defending_q[i] = float(np.mean(d_vals)) if d_vals else 0.0

    X = sparse.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )

    if pass1_quality is not None:
        q_mat = sparse.csr_matrix(
            np.column_stack([acting_q, defending_q]).astype(np.float32)
        )
        X = sparse.hstack([X, q_mat], format="csr")

    return X

# ── Fitting functions ────────────────────────────────────────────────────────

def fit_rapm_raw(frame, X, skater_ids, goalie_list_arg, sample_weight=None):
    """Standard RidgeCV fit (no prior). Returns results dict, coefs dict, alphas, and goalie results."""
    results        = {pid: {} for pid in skater_ids}
    goalie_results = {gid: {} for gid in goalie_list_arg}
    all_coefs      = {}
    alphas         = {}
    goalie_offset  = len(skater_ids) * 2

    for metric_name, y_col in METRICS.items():
        y = frame[y_col].fillna(0).astype(float).values
        model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
        model.fit(X, y, **({"sample_weight": sample_weight} if sample_weight is not None else {}))

        alpha = model.alpha_
        coefs = model.coef_
        all_coefs[metric_name] = coefs
        alphas[metric_name] = float(alpha)

        n_pos = int((y > 0).sum())
        print(f"    {metric_name}: {n_pos:,} events y>0 | alpha={alpha:.0f}", file=sys.stderr)

        scale = EVENTS_PER_60
        for i, pid in enumerate(skater_ids):
            o    = float(coefs[i * 2])     * scale
            d    = float(coefs[i * 2 + 1]) * scale
            results[pid][f"{metric_name}_O"]   = round(o, 4)
            results[pid][f"{metric_name}_D"]   = round(d, 4)
            results[pid][f"{metric_name}_net"] = round(o - d, 4)

        # Extract goalie coefficients
        for gi, gid in enumerate(goalie_list_arg):
            g_coef = float(coefs[goalie_offset + gi]) * scale
            goalie_results[gid][f"{metric_name}_G"] = round(g_coef, 4)

    # BPR composite
    for pid in skater_ids:
        r = results[pid]
        bpr_o = sum(BPR_WEIGHTS[m][0] * r[f"{m}_O"] for m in METRICS)
        bpr_d = sum(BPR_WEIGHTS[m][1] * r[f"{m}_D"] for m in METRICS)
        r["BPR_O"] = round(bpr_o, 4)
        r["BPR_D"] = round(bpr_d, 4)
        r["BPR"]   = round(bpr_o + bpr_d, 4)

    return results, all_coefs, alphas, goalie_results


def fit_rapm_prior(frame, X, skater_ids, goalie_list_arg, prior_lookup, calibration,
                   sample_weight=None, raw_alphas=None):
    """Prior-informed ridge fit. Uses raw RAPM alphas with prior mean shift."""
    results        = {pid: {} for pid in skater_ids}
    goalie_results = {gid: {} for gid in goalie_list_arg}
    all_coefs      = {}
    n_cols         = X.shape[1]
    goalie_offset  = len(skater_ids) * 2

    cal_O = calibration["offense"]
    cal_D = calibration["defense"]

    # Total BPR weight for proportional decomposition
    total_O_weight = sum(abs(w[0]) for w in BPR_WEIGHTS.values())
    total_D_weight = sum(abs(w[1]) for w in BPR_WEIGHTS.values())

    # Prior SD in per-event units (for posterior SE calculation)
    prior_sd_o_full = cal_O["prior_sd"] / EVENTS_PER_60
    prior_sd_d_full = cal_D["prior_sd"] / EVENTS_PER_60

    for metric_name, y_col in METRICS.items():
        y = frame[y_col].fillna(0).astype(float).values

        # Build prior mean vector
        mu = np.zeros(n_cols, dtype=np.float64)
        w_o = BPR_WEIGHTS[metric_name][0] / total_O_weight
        w_d = BPR_WEIGHTS[metric_name][1] / total_D_weight

        for i, pid in enumerate(skater_ids):
            if pid in prior_lookup:
                prior_O, prior_D = prior_lookup[pid]
                mu[i * 2]     = (prior_O * w_o) / EVENTS_PER_60
                mu[i * 2 + 1] = (prior_D * w_d) / EVENTS_PER_60

        # Use raw RAPM's RidgeCV alpha if available, otherwise use RidgeCV
        if raw_alphas and metric_name in raw_alphas:
            alpha = raw_alphas[metric_name]
        else:
            # Fallback: run RidgeCV on unshifted data to get alpha
            cv_model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
            cv_model.fit(X, y, **({"sample_weight": sample_weight} if sample_weight is not None else {}))
            alpha = cv_model.alpha_

        # Shift target and fit with prior mean
        y_adj = y - X.dot(mu)
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y_adj, **({"sample_weight": sample_weight} if sample_weight is not None else {}))

        coefs = model.coef_ + mu
        all_coefs[metric_name] = coefs

        n_pos = int((y > 0).sum())
        print(f"    {metric_name}: {n_pos:,} events y>0 | alpha={alpha:.0f}", file=sys.stderr)

        # Extract skater coefficients + posterior SE
        scale = EVENTS_PER_60
        residuals = y - X.dot(coefs)
        sigma2 = float(np.mean(residuals**2))

        col_counts = np.array(np.abs(X).sum(axis=0)).flatten()

        # Per-metric prior SD in per-event units
        prior_sd_o_m = prior_sd_o_full * abs(w_o)
        prior_sd_d_m = prior_sd_d_full * abs(w_d)

        for i, pid in enumerate(skater_ids):
            o    = float(coefs[i * 2])     * scale
            d    = float(coefs[i * 2 + 1]) * scale

            # Posterior SE: 1/(n/sigma2 + 1/prior_var)
            n_o = max(float(col_counts[i * 2]), 1.0)
            n_d = max(float(col_counts[i * 2 + 1]), 1.0)
            post_var_o = 1.0 / (n_o / sigma2 + 1.0 / max(prior_sd_o_m**2, 1e-10))
            post_var_d = 1.0 / (n_d / sigma2 + 1.0 / max(prior_sd_d_m**2, 1e-10))
            o_se = np.sqrt(post_var_o) * scale
            d_se = np.sqrt(post_var_d) * scale

            results[pid][f"{metric_name}_O"]    = round(o, 4)
            results[pid][f"{metric_name}_D"]    = round(d, 4)
            results[pid][f"{metric_name}_net"]  = round(o - d, 4)
            results[pid][f"{metric_name}_O_se"] = round(o_se, 4)
            results[pid][f"{metric_name}_D_se"] = round(d_se, 4)

        # Extract goalie coefficients
        for gi, gid in enumerate(goalie_list_arg):
            g_coef = float(coefs[goalie_offset + gi]) * scale
            goalie_results[gid][f"{metric_name}_G"] = round(g_coef, 4)

    # BPR composite + propagated uncertainty
    for pid in skater_ids:
        r = results[pid]
        bpr_o = sum(BPR_WEIGHTS[m][0] * r[f"{m}_O"] for m in METRICS)
        bpr_d = sum(BPR_WEIGHTS[m][1] * r[f"{m}_D"] for m in METRICS)
        r["BPR_O"] = round(bpr_o, 4)
        r["BPR_D"] = round(bpr_d, 4)
        r["BPR"]   = round(bpr_o + bpr_d, 4)

        if MODE == "prior":
            bpr_o_se = np.sqrt(sum((BPR_WEIGHTS[m][0] * r.get(f"{m}_O_se", 0))**2 for m in METRICS))
            bpr_d_se = np.sqrt(sum((BPR_WEIGHTS[m][1] * r.get(f"{m}_D_se", 0))**2 for m in METRICS))
            r["BPR_O_se"] = round(bpr_o_se, 4)
            r["BPR_D_se"] = round(bpr_d_se, 4)
            r["BPR_se"]   = round(np.sqrt(bpr_o_se**2 + bpr_d_se**2), 4)

    return results, all_coefs, goalie_results


def results_to_df(results, season=None):
    out = pd.DataFrame.from_dict(results, orient="index")
    out.index.name = "player_id"
    out = out.reset_index()
    out = out.merge(player_names, on="player_id", how="left")
    out["position"] = out["player_id"].map(pid_to_pos).fillna("?")
    if season is not None:
        out.insert(1, "season", season)
    return out.sort_values("BPR", ascending=False)

# ── Build indices ────────────────────────────────────────────────────────────
print("\nBuilding player and goalie indices...", file=sys.stderr)
skater_ids = build_player_index(df)
skater_idx = {pid: i for i, pid in enumerate(skater_ids)}
goalie_list = build_goalie_index(df)
goalie_idx  = {gid: i for i, gid in enumerate(goalie_list)}
print(f"  {len(skater_ids):,} skaters, {len(goalie_list)} goalies", file=sys.stderr)

# ── Build design matrix ─────────────────────────────────────────────────────
print("Building design matrix...", file=sys.stderr)
X = build_design_matrix(df, skater_ids, skater_idx, goalie_list, goalie_idx)
sample_weight = df["recency_weight"].values
print(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}, {X.nnz:,} nnz", file=sys.stderr)

# ── Fit model ────────────────────────────────────────────────────────────────
if MODE == "raw":
    print("\n── Pooled raw RAPM (no prior) ──────────────", file=sys.stderr)
    results, all_coefs, raw_alphas, g_results = fit_rapm_raw(
        df, X, skater_ids, goalie_list, sample_weight=sample_weight,
    )
    out_df = results_to_df(results)

    # Save raw alphas for prior mode to use
    with open("data/v2_raw_alphas.json", "w") as f:
        json.dump(raw_alphas, f, indent=2)
    print(f"  Saved per-metric alphas: {raw_alphas}", file=sys.stderr)

    # Save goalie RAPM
    g_df = pd.DataFrame.from_dict(g_results, orient="index")
    g_df.index.name = "goalie_id"
    g_df = g_df.reset_index()
    g_df["goalie_name"] = g_df["goalie_id"].map(goalie_names).fillna("?")
    g_df.to_csv("data/v2_goalie_rapm.csv", index=False)
    print(f"  Saved {len(g_df)} goalie coefficients to data/v2_goalie_rapm.csv", file=sys.stderr)

    out_df.to_csv("data/v2_rapm_raw.csv", index=False)
    print(f"\nWrote {len(out_df):,} skaters to data/v2_rapm_raw.csv", file=sys.stderr)

    cols = ["player_name", "position", "BPR_O", "BPR_D", "BPR",
            "xGF_net", "GF_net", "SOG_net", "TO_net", "GA_net"]
    print("\nTop 15 (raw BPR):", file=sys.stderr)
    print(out_df.head(15)[cols].to_string(index=False), file=sys.stderr)

elif MODE == "prior":
    # Load calibration and prior
    print("\nLoading prior calibration and box score priors...", file=sys.stderr)
    with open("data/v2_prior_calibration.json") as f:
        calibration = json.load(f)
    print(f"  Calibration: O r2_eff={calibration['offense']['r2_eff']}, "
          f"D r2_eff={calibration['defense']['r2_eff']}", file=sys.stderr)

    # Load raw RAPM alphas (from --mode=raw run)
    raw_alphas = None
    raw_alphas_file = Path("data/v2_raw_alphas.json")
    if raw_alphas_file.exists():
        with open(raw_alphas_file) as f:
            raw_alphas = json.load(f)
        print(f"  Using raw RAPM alphas: {raw_alphas}", file=sys.stderr)

    prior_df = pd.read_csv("data/v2_box_prior.csv")
    prior_df["player_id"] = prior_df["player_id"].astype(int)

    # Build pooled prior: average across seasons per player
    prior_pooled = prior_df.groupby("player_id")[["prior_O", "prior_D"]].mean()
    prior_lookup = {
        int(pid): (float(row["prior_O"]), float(row["prior_D"]))
        for pid, row in prior_pooled.iterrows()
    }
    print(f"  {len(prior_lookup):,} players with box score priors", file=sys.stderr)

    # ── Pooled fit ───────────────────────────────────────────────────────────
    print("\n── Pooled prior-informed RAPM ──────────────", file=sys.stderr)
    results, all_coefs, g_results = fit_rapm_prior(
        df, X, skater_ids, goalie_list,
        prior_lookup, calibration,
        sample_weight=sample_weight,
        raw_alphas=raw_alphas,
    )
    out_df = results_to_df(results)

    # Save goalie RAPM
    g_df = pd.DataFrame.from_dict(g_results, orient="index")
    g_df.index.name = "goalie_id"
    g_df = g_df.reset_index()
    g_df["goalie_name"] = g_df["goalie_id"].map(goalie_names).fillna("?")
    g_df.to_csv("data/v2_goalie_rapm.csv", index=False)
    print(f"  Saved {len(g_df)} goalie coefficients to data/v2_goalie_rapm.csv", file=sys.stderr)

    # Add prior columns for diagnostics
    out_df["prior_O"] = out_df["player_id"].map(lambda p: prior_lookup.get(p, (0, 0))[0]).round(4)
    out_df["prior_D"] = out_df["player_id"].map(lambda p: prior_lookup.get(p, (0, 0))[1]).round(4)

    out_df.to_csv("data/v2_rapm_results.csv", index=False)
    print(f"\nWrote {len(out_df):,} skaters to data/v2_rapm_results.csv", file=sys.stderr)

    cols = ["player_name", "position", "BPR_O", "BPR_D", "BPR", "BPR_se",
            "prior_O", "prior_D"]
    print("\nTop 15 (prior-informed BPR):", file=sys.stderr)
    print(out_df.head(15)[cols].to_string(index=False), file=sys.stderr)

    # ── Per-season fits ──────────────────────────────────────────────────────
    # Build pass-1 quality lookup
    pass1_quality = {
        int(row["player_id"]): (float(row["BPR_O"]), float(row["BPR_D"]))
        for _, row in out_df.iterrows()
    }

    season_frames = []
    for season in sorted(df["season"].unique()):
        sf = df[df["season"] == season].reset_index(drop=True)
        if len(sf) < 10_000:
            print(f"  Skipping season {season} ({len(sf):,} events)", file=sys.stderr)
            continue

        print(f"\n── Season {season} ({len(sf):,} events) ──────────────", file=sys.stderr)
        s_skaters = build_player_index(sf)
        s_skater_idx = {pid: i for i, pid in enumerate(s_skaters)}
        s_goalies = build_goalie_index(sf)
        s_goalie_idx = {gid: i for i, gid in enumerate(s_goalies)}

        sX = build_design_matrix(sf, s_skaters, s_skater_idx, s_goalies, s_goalie_idx,
                                  pass1_quality=pass1_quality)
        print(f"  {len(s_skaters)} skaters, {len(s_goalies)} goalies, "
              f"matrix {sX.shape[0]}×{sX.shape[1]}", file=sys.stderr)

        # Per-season prior: use season-specific box score if available, else pooled
        s_prior = prior_df[prior_df["season"] == season].set_index("player_id")
        s_prior_lookup = {}
        for pid in s_skaters:
            if pid in s_prior.index:
                s_prior_lookup[pid] = (
                    float(s_prior.loc[pid, "prior_O"]),
                    float(s_prior.loc[pid, "prior_D"]),
                )
            elif pid in prior_lookup:
                s_prior_lookup[pid] = prior_lookup[pid]

        s_results, _, _ = fit_rapm_prior(
            sf, sX, s_skaters, s_goalies,
            s_prior_lookup, calibration,
            raw_alphas=raw_alphas,
        )
        s_df = results_to_df(s_results, season=season)
        s_df["prior_O"] = s_df["player_id"].map(lambda p: s_prior_lookup.get(p, (0, 0))[0]).round(4)
        s_df["prior_D"] = s_df["player_id"].map(lambda p: s_prior_lookup.get(p, (0, 0))[1]).round(4)
        season_frames.append(s_df)

    if season_frames:
        all_seasons = pd.concat(season_frames, ignore_index=True)
        all_seasons.to_csv("data/v2_rapm_by_season.csv", index=False)
        print(f"\nWrote {len(all_seasons):,} player-season rows to data/v2_rapm_by_season.csv",
              file=sys.stderr)
