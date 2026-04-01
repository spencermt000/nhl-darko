---
title: Surplus Value — What Players Are Really Worth (v2)
date: 2026-03-31
---

# Surplus Value — What Players Are Really Worth

Every NHL contract is a bet: the team pays a fixed cap hit hoping the player's production justifies the cost. Surplus value measures whether that bet paid off.

## The Model (v2 — Multi-Metric)

Our current surplus model uses a **Ridge regression trained on all player-seasons** where the target is the player's actual Cap %. The model learns what the market pays for different stat profiles.

**Features:**
- Per-game rates: goals, assists, points, shots, hits, blocks
- Ice time: TOI/GP, PP usage, PK usage
- Advanced metrics: WAR, WAR_82, GAR components (EV_O, EV_D, PP, PK, PEN)
- Per-60 rates: EV_O_rate, EV_D_rate, PP_rate, PK_rate
- BPR components: BPR_O, BPR_D, total_BPR, PP_O, PK_D
- Player context: age, age², position, contract type (ELC), UFA status, draft round, pro years

**Model performance:** R² = 0.67, MAE = 1.36% of cap

The model predicts what Cap % a player's stats are worth → multiply by the salary cap to get dollar market value.

```
Predicted Market Value = pred_cap_pct / 100 × salary_cap
Surplus = Predicted Market Value - Actual Cap Hit
```

## What the Model Learned

The top factors driving cap hit:

| Feature | Effect | Meaning |
|---------|--------|---------|
| PP_O | +2.06% per unit | PP production is the most valued skill |
| WAR_82 | +1.21% per unit | Pace-adjusted WAR matters |
| ELC | -1.08% | ELCs are ~1.1% of cap cheaper than stats warrant |
| Position (D) | -1.05% | Defensemen earn less for equivalent stats |
| 1st Round Pick | +0.87% | Draft pedigree commands a premium |
| Age | +0.78%/yr | Older players cost more (reputation premium) |
| UFA | +0.42% | Free agents pay ~0.4% more for same production |

## Key Findings

**By contract type:**
- **ELCs** average +1.91% of cap surplus per season — the best value in hockey
- **RFAs** average ~0% surplus — near fair value in the restricted market
- **UFAs** average -0.56% — slightly overpaid (free market premium)

**Best value contracts (2025-26):**

| Player | Cap Hit | Market Value | Surplus |
|--------|---------|-------------|---------|
| Cale Makar | $881K (ELC) | $7.1M | +$6.2M |
| Sebastian Aho | $775K | $6.3M | +$5.6M |
| Adam Fox | $925K (ELC) | $5.4M | +$4.5M |

**Worst value contracts (2025-26):**

| Player | Cap Hit | Market Value | Surplus |
|--------|---------|-------------|---------|
| Elias Pettersson | $11.6M | $5.7M | -$5.9M |
| Jeff Skinner | $9.0M | $3.8M | -$5.2M |
| Drew Doughty | $11.0M | $5.9M | -$5.1M |

## Why This Approach Works Better Than $/WAR

Our v1 model used a single `$/WAR` multiplier ($17.5M per WAR from UFA market calibration). The problem: WAR is one-dimensional. Two players with the same WAR but different stat profiles (one a goal scorer, the other a defensive specialist) would get the same market value.

The v2 Ridge model captures nuance:
- The market overpays for point production relative to defensive value
- Older players get paid more for the same stats (reputation/track record)
- First-round picks command a premium even years after being drafted
- ELC players are systematically underpaid (that's the design of ELCs)
- UFAs command a premium vs RFAs for identical production

## How We Use Surplus

Surplus is the foundation for:
- **Trade evaluation** — comparing what you're giving up vs getting back
- **Roster construction** — finding where to allocate cap space
- **Draft strategy** — ELC surplus is why draft picks are valuable
- **Contract negotiation** — what a player's stats say they're worth vs what they're asking
