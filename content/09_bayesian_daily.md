---
title: How Daily Ratings Work — A DARKO-Style Approach
date: 2026-03-31
---

# How Daily Ratings Work — A DARKO-Style Approach

Inspired by the DARKO system in basketball, our daily ratings update after every game using Bayesian smoothing. This creates a real-time player evaluation that balances recent performance with historical priors.

## The 5 Components

Each player has 5 independent rating components, all measured in goals above average per 60 minutes:

| Component | What It Captures | Prior SE | Evidence Weight |
|-----------|-----------------|----------|-----------------|
| **EV_O** | 5v5 offensive impact | 0.08 | High (15 min ref TOI) |
| **EV_D** | 5v5 defensive impact | 0.05 | High (15 min ref TOI) |
| **PP** | Power play production | 0.15 | Low (2 min ref TOI) |
| **PK** | Penalty kill defense | 0.10 | Low (2 min ref TOI) |
| **PEN** | Penalty drawing/taking | 0.05 | High (15 min ref TOI) |

## How the Update Works

After each game, each component is updated:

```
posterior = (prior_precision × prior + evidence_weight × observation) / total_precision
```

- **Prior** = the player's pre-game rating (accumulated from all previous games)
- **Observation** = what happened in this specific game
- **Evidence weight** = how much TOI the player logged (more ice time = more evidence)
- **Precision** = inverse of uncertainty (higher = more confident)

## Key Parameters

**Decay halflife: 30 games.** After 30 games without playing, a player's evidence weight decays by 50%. This means recent performance matters more than games from 3 months ago.

**Offseason carry: 85%.** Between seasons, 85% of the prior carries forward. A player who was elite last year starts this year with a strong prior, but it's not locked in — they need to re-earn it.

**Prior sources:** EV_O and EV_D priors come from the carry-forward composite (the predictive model). PP and PK priors come from career RAPM estimates.

## What Makes This Different

Traditional stats (points, goals) are cumulative — they only go up. Our daily ratings go **both up and down** based on each game's contribution. A player who has a terrible stretch will see their rating drop in real-time, even if their season totals still look fine.

This is especially useful for:
- **Identifying hot/cold streaks** before they show up in traditional stats
- **Evaluating mid-season trades** — did a player improve after changing teams?
- **Injury impact** — how does a player's rating change when a key teammate goes down?

## Converting to GAR

The daily ratings (per-60 rates) convert to Goals Above Replacement by multiplying by ice time:

```
EV_O_GAR = sum over all games of (EV_O_posterior × toi_5v5 / 60)
PP_GAR = sum over all games of (PP_posterior × toi_pp / 60)
...
GAR = EV_O_GAR + EV_D_GAR + PP_GAR + PK_GAR + PEN_GAR
WAR = GAR / 6.0
```

This is how a single per-60 rate turns into a counting stat that represents total seasonal value.
