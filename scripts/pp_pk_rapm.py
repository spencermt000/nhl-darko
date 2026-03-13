"""
pp_pk_rapm.py — Power play and penalty kill RAPM ratings.

Single unified model on all unequal-strength events (5v4, 4v5, 5v3, 3v5).
Design convention:
  - PP team (more skaters): O columns = +1
  - PK team (fewer skaters): D columns = -1

y = 1 if the PP team generated the event (shot/goal), 0 if PK team did.
Ridge regression then estimates:
  PP_O: player contribution to PP shot/goal generation
  PK_D: player contribution to PK shot/goal prevention (positive = good)

Outputs:
  data/pp_rapm.csv  — per player PP_O, PK_D (from O and D columns respectively)
  data/pk_rapm.csv  — same file, different view (sorted by PK_D)

Note: One model, one output. pp_rapm.csv has both components.
"""

import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV
from pathlib import Path

CLEAN_PBP   = Path("data/clean_pbp.csv")
RAPM_POOLED = Path("data/rapm_results.csv")
BIO_FILE    = Path("data/moneypuck_player_bio.csv")
OUT_PP      = Path("data/pp_rapm.csv")

# Unequal-strength states where PP team (more skaters) is on offense
PP_STRENGTHS = {"5v4", "4v5", "5v3", "3v5"}

METRICS = {
    "xGF": "xGoal",
    "GF":  "is_goal",
}

# For PP, weight offense heavily (2/3) and defense (1/3)
PP_WEIGHTS  = {"xGF": (0.75, 0.25), "GF": (0.75, 0.25)}

# Very strong regularization — PP data is sparse and collinear
ALPHA_CANDIDATES = [5000, 10000, 20000, 50000, 100000]
EVENTS_PER_60    = 90.0

# Minimum column activations to include a player
MIN_EVENTS = 300   # ~3 hours of PP or PK ice time — cuts noise from fringe/retired players

# ── Goalie exclusion ──────────────────────────────────────────────────────────
print("Identifying goalies...", file=sys.stderr)
goalie_ids = set(
    pd.read_csv("data/raw_pbp.csv", usecols=["event_goalie_id"])
    ["event_goalie_id"].dropna().astype(float).astype(int)
)
print(f"  {len(goalie_ids)} goalie IDs to exclude", file=sys.stderr)

# ── Position lookup ───────────────────────────────────────────────────────────
bio = pd.read_csv(BIO_FILE).rename(columns={"playerId": "player_id"})
bio["player_id"] = bio["player_id"].astype(int)
def norm_pos(p):
    p = str(p).upper()
    return "F" if p in ("C","L","R","LW","RW","F") else ("D" if p=="D" else "F")
pid_to_pos = bio.set_index("player_id")["position"].map(norm_pos).to_dict()

# ── 5v5 RAPM for player name lookup ──────────────────────────────────────────
rapm5 = pd.read_csv(RAPM_POOLED)
rapm5["player_id"] = rapm5["player_id"].astype(int)

# ── Load PBP ──────────────────────────────────────────────────────────────────
print("Loading clean_pbp.csv...", file=sys.stderr)
df = pd.read_csv(CLEAN_PBP)

# Filter to unequal-strength events
pp_df = df[df["strength"].isin(PP_STRENGTHS)].copy().reset_index(drop=True)
print(f"  PP/PK events: {len(pp_df):,}", file=sys.stderr)

# ── Determine PP team and PK team per event ───────────────────────────────────
# strength column = "home_skaters v away_skaters"
# PP team = side with MORE skaters
home_skaters = pp_df["strength"].str.split("v").str[0].astype(int)
away_skaters = pp_df["strength"].str.split("v").str[1].astype(int)
home_is_pp   = (home_skaters > away_skaters).values

# For each event: pp_ice = on-ice string for PP team, pk_ice = PK team
pp_df["pp_ice"] = np.where(home_is_pp, pp_df["home_on_ice"], pp_df["away_on_ice"])
pp_df["pk_ice"] = np.where(home_is_pp, pp_df["away_on_ice"], pp_df["home_on_ice"])

# y = 1 if PP team generated the event, 0 if PK team did
pp_team_acting = (
    ((pp_df["event_team_type"] == "home") & home_is_pp) |
    ((pp_df["event_team_type"] == "away") & ~home_is_pp)
)
pp_df["pp_acting"] = pp_team_acting.values

# ── Helper functions ──────────────────────────────────────────────────────────
def parse_ids(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        return []

def build_player_index(frame):
    ids = set()
    for col in ("pp_ice", "pk_ice"):
        for s in frame[col]:
            ids.update(parse_ids(s))
    return sorted(ids - goalie_ids)

def build_design_matrix(frame, player_ids, pid_to_idx):
    """
    PP players (pp_ice): O columns = +1
    PK players (pk_ice): D columns = -1
    """
    n_events  = len(frame)
    n_cols    = len(player_ids) * 2
    rows, cols_out, vals = [], [], []

    pp_ice_arr = frame["pp_ice"].values
    pk_ice_arr = frame["pk_ice"].values

    for i in range(n_events):
        for pid in parse_ids(pp_ice_arr[i]):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2);     vals.append(1.0)
        for pid in parse_ids(pk_ice_arr[i]):
            if pid in pid_to_idx:
                rows.append(i); cols_out.append(pid_to_idx[pid] * 2 + 1); vals.append(-1.0)

    return sparse.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols_out, dtype=np.int32))),
        shape=(n_events, n_cols),
    )

# ── Build player index and design matrix ──────────────────────────────────────
print("\n── PP/PK model ──────────────────────────────────────────", file=sys.stderr)
player_ids = build_player_index(pp_df)
pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
print(f"  {len(player_ids):,} skaters", file=sys.stderr)

X = build_design_matrix(pp_df, player_ids, pid_to_idx)
print(f"  Matrix: {X.shape[0]:,} × {X.shape[1]:,}, {X.nnz:,} nnz", file=sys.stderr)

# Filter to players with MIN_EVENTS activations
col_counts = np.array(np.abs(X).sum(axis=0)).flatten()
keep_mask  = np.array([(col_counts[i*2] + col_counts[i*2+1]) >= MIN_EVENTS
                        for i in range(len(player_ids))])
player_ids = [pid for pid, k in zip(player_ids, keep_mask) if k]
pid_to_idx = {pid: i for i, pid in enumerate(player_ids)}
keep_cols  = [j for i, k in enumerate(keep_mask) if k for j in (i*2, i*2+1)]
X = X[:, keep_cols]
print(f"  After MIN_EVENTS={MIN_EVENTS} filter: {len(player_ids):,} skaters", file=sys.stderr)

# ── Fit metrics ───────────────────────────────────────────────────────────────
# No prior shifting — standard ridge shrinkage toward zero.
# The 5v5 RAPM prior is in per-60 units but model operates in per-event units;
# mismatched scales make the prior counterproductive for the PP model.
results = {pid: {} for pid in player_ids}

for m, y_col in METRICS.items():
    # y = positive when PP team generates, negative when PK team generates
    y        = pp_df[y_col].fillna(0).astype(float).values
    y_signed = np.where(pp_df["pp_acting"].values, y, -y)

    model = RidgeCV(alphas=ALPHA_CANDIDATES, fit_intercept=True)
    model.fit(X, y_signed)
    coefs = model.coef_

    n_pos = int((y_signed > 0).sum())
    print(f"  {m}: {n_pos:,} PP events | alpha={model.alpha_:.0f}", file=sys.stderr)

    for i, pid in enumerate(player_ids):
        results[pid][f"{m}_O"] = round(float(coefs[i*2])   * EVENTS_PER_60, 4)
        results[pid][f"{m}_D"] = round(float(coefs[i*2+1]) * EVENTS_PER_60, 4)

# ── Composite ratings ─────────────────────────────────────────────────────────
for pid in player_ids:
    r = results[pid]
    # PP_BPR = O-weighted (PP specialists are measured on offense)
    r["PP_O"]   = round(sum(PP_WEIGHTS[m][0] * r[f"{m}_O"] for m in METRICS), 4)
    r["PK_D"]   = round(sum(PP_WEIGHTS[m][1] * r[f"{m}_D"] for m in METRICS), 4)
    r["PP_BPR"] = round(r["PP_O"] + r["PK_D"], 4)  # combined two-way special teams

# ── Output ────────────────────────────────────────────────────────────────────
out = pd.DataFrame.from_dict(results, orient="index")
out.index.name = "player_id"
out = out.reset_index()
out = out.merge(
    rapm5[["player_id","player_name"]].drop_duplicates("player_id"),
    on="player_id", how="left"
)
out["position"] = out["player_id"].map(pid_to_pos).fillna("?")

out.to_csv(OUT_PP, index=False)
print(f"\nWrote {len(out):,} players → {OUT_PP}", file=sys.stderr)

print("\nTop 20 PP specialists (PP_O):")
print(out.sort_values("PP_O", ascending=False).head(20)
      [["player_name","position","PP_O","PK_D","PP_BPR"]].to_string(index=False))

print("\nTop 20 PK specialists (PK_D):")
print(out.sort_values("PK_D", ascending=False).head(20)
      [["player_name","position","PP_O","PK_D","PP_BPR"]].to_string(index=False))
