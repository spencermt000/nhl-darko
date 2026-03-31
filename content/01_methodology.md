---
title: How the Model Works
date: 2026-03-30
---

# How the Model Works

This player rating system combines multiple independent signal sources — on-ice regression (RAPM), box score production models, and expected performance models — into a single predictive composite, then smooths across seasons with a Bayesian carry-forward.

## The Building Blocks

### RAPM (Regularized Adjusted Plus/Minus)
Every event on ice becomes a regression row. Every player becomes a column (+1 when their team acts, -1 when defending). Ridge regression isolates each player's individual contribution, controlling for every teammate and opponent simultaneously. We run this on 3-season rolling windows to capture current talent level.

### Goal Value (GV)
A box-score production model: Ridge regression from ~25 individual stats (xGoals, goals, assists, shots, blocks, takeaways, zone deployment) predicting on-ice goal generation and suppression.

### On/Off Impact (OOI)
Same box-score features, but predicting a player's on-ice vs off-ice xGF differential. Captures system effects that individual counting stats miss.

## Blending Into a Composite

Two predictive layers combine these signals:

- **Production Value (PV):** Predicts next-season individual production. GV dominates — production is stable and self-predictive.
- **Impact Value (IV):** Predicts next-season on-ice impact. OOI and RAPM dominate — context predicts future system impact.

The final composite blends both 50/50.

## Bayesian Carry-Forward

The key innovation. Without carry-forward, each season resets to zero. With it, ratings accumulate career information:

- **85% carry** from prior season (modified by age)
- **Sigmoid blend** between current data and carried-forward prior
- Young players (< 22) carry less (they change fast); peak-age players carry more

This more than doubled year-over-year prediction stability (r² from 0.27 → 0.56).

## GAR/WAR

Goals Above Replacement / Wins Above Replacement with 5 components:
- Even-strength offense & defense
- Power play & penalty kill
- Penalties (drawing vs taking)

Replacement level is set at the 17th percentile. 6 goals = 1 win.
