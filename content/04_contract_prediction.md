---
title: Predicting NHL Contracts
date: 2026-03-31
---

# Predicting NHL Contracts

Can we predict what a player will earn on their next contract? We trained an XGBoost model on ~1,000 historical standard contracts to find out.

## What the Model Learned

The top features driving contract value, in order of importance:

1. **Points per game** (0.24) — the single biggest driver. Scoring sells.
2. **2-year WAR average** (0.18) — sustained value matters more than one hot season.
3. **Shots per game** (0.12) — volume shooters get paid, even controlling for goals.
4. **Blocks per game** (0.06) — defensive engagement signals are valued.
5. **Position: defenseman** (0.05) — D-men earn differently than forwards for equivalent stats.
6. **Age at signing** (0.03) — older players get shorter, cheaper deals.
7. **PP TOI per game** (0.02) — power play usage = trust = money.

## Model Performance

| Target | Test MAE | Test R² |
|--------|----------|---------|
| Cap % | 0.99% | 0.773 |
| Term | 1.09 yr | 0.510 |

The model predicts cap hit percentage within ~1% of the actual salary cap on average. Term is harder — it's more about negotiation leverage than pure performance.

## UFA vs RFA Market

The model captures the structural difference between UFA and RFA contracts. UFAs get paid ~0.4% of cap more than RFAs with identical stats. This "UFA premium" reflects the leverage of unrestricted free agency.

## Where the Model Struggles

- **Record-breaking deals** (Kaprizov's $17M, McDavid's extensions) — the model hasn't seen deals at this scale before
- **Reputation vs production** — veterans with big names but declining stats (Stamkos, Ovechkin) get paid above what the model expects
- **Bridge deals** — short-term RFA contracts are harder to predict because they're strategic, not purely market-driven

## Confidence Intervals

We report 80% prediction intervals computed from cross-validated residuals. For a typical projection:
- AAV: ±$1.5M (±1.5% of cap)
- Term: ±1.5 years
