---
title: Win Shares — Allocating Actual Wins
date: 2026-03-30
---

# Win Shares — Allocating Actual Wins

WAR tells you how valuable a player is above replacement. But it doesn't answer a simpler question: **how many of my team's wins did this player actually account for?**

Win Shares answers that. It takes a team's actual win total and distributes it to players proportionally based on their GAR contributions.

## How It Works

For each team-season:

1. Take every player's offensive GAR (GAR_O) and defensive GAR (GAR_D)
2. Floor each at zero — bad players get 0 win shares, not negative
3. Sum the floored values across the roster
4. Each player's share of team wins = their floored GAR / team total floored GAR × team wins

This guarantees three properties:
- **Non-negative:** No player can have negative win shares
- **Sums to team wins:** All player win shares on a team add up to exactly the team's actual wins
- **Context-dependent:** A 2-WAR player on a 50-win team gets more win shares than the same player on a 30-win team

## WS vs WAR

Win Shares and WAR are correlated (r ≈ 0.77) but distinct. WAR measures individual value in a vacuum. Win Shares measures contribution within a team context. Both are useful — WAR for player evaluation, WS for understanding how wins actually happened.

## Key Insight

The biggest divergences between WS and WAR happen for elite players on bad teams. A star putting up 3.0 WAR on a 25-win team gets fewer win shares than a 2.0 WAR player on a 55-win team. That's not a bug — it reflects that wins are a team outcome.
