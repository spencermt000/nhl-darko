---
title: NHL Age Curves — When Do Players Peak?
date: 2026-03-31
---

# NHL Age Curves — When Do Players Peak?

Using 10 seasons of carry-forward data (7,500+ player-seasons), we built empirical age curves showing how player value evolves over a career.

## WAR by Age

The average WAR/82 (pace-adjusted to a full season) by age:

| Age | WAR/82 (% of peak) | Description |
|-----|---------------------|-------------|
| 20 | 84% | Rookie adjustment period |
| 22 | 91% | Rapid development |
| 24 | 93% | Approaching prime |
| 26 | 101% | Entering peak |
| 28 | 106% | **Peak performance** |
| 30 | 107% | Still elite |
| 32 | 101% | Beginning of plateau |
| 34 | 100% | Slow decline starts |
| 36 | 93% | Noticeable drop |
| 38 | 95% | Survivorship bias (only the best are still playing) |
| 40 | 63% | Steep late-career decline |

## Key Takeaways

**Peak is 28-30, not 25-27.** Contrary to popular belief, NHL skaters peak around age 28-30 in our data. This is later than often cited because our WAR metric captures defense and special teams, not just offensive production. Offense may peak earlier, but overall value (including defensive awareness, penalty kill, and positional play) peaks later.

**The decline is gradual until 35.** Players retain 93-100% of their peak value through age 36. The cliff doesn't hit until 38+.

**Survivorship bias after 36.** The apparent plateau at 38 (95%) is misleading — only elite players like Crosby, Ovechkin, and Bergeron are still in the league at that age. The average player who was in the league at 34 has already retired by 38.

**Young players are underrated.** A 20-year-old producing at 84% of peak is actually outperforming expectations for their age. If you see a 20-year-old putting up average numbers, they're likely to improve significantly.

## Implications for Contracts

The age curve directly feeds our contract NPV model. A long-term deal signed at age 25 captures the full peak years (26-30) plus the gradual decline. A deal signed at 32 starts strong but declines faster.

**Best contract timing:** Sign players at 24-25 on long-term deals to capture the peak. Avoid 8-year deals starting at 30+ — the back half will be underwater.

## How We Use This

The age curve modifies our player projections:
- **Under 25:** Apply a 1.03-1.10x multiplier (still developing)
- **25-30:** Apply 1.00x (peak, no adjustment needed)
- **31-33:** Apply 0.97x (gradual decline)
- **34+:** Apply 0.88-0.93x (meaningful decline)
