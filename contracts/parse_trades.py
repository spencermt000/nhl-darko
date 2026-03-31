"""
Parse trade HTML files from prosportstransactions.com into structured CSV.

Reads HTML files from trades/ directory and extracts:
  - Trade date/teams
  - Players and picks exchanged on each side

Usage: python contracts/parse_trades.py
"""

import os
import re
import pandas as pd
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_DIR = os.path.join(BASE, "trades")

all_trades = []
trade_id = 0

for fname in sorted(os.listdir(TRADES_DIR)):
    if not fname.endswith(".html"):
        continue

    year_match = re.search(r"(\d{4})", fname)
    if not year_match:
        continue
    draft_year = int(year_match.group(1))

    with open(os.path.join(TRADES_DIR, fname)) as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table")
    if not table:
        continue

    rows = table.find_all("tr")
    print(f"\n{fname}: {len(rows)} rows")

    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 5:
            continue

        transactions = cells[3].get_text(strip=True)
        if not transactions:
            continue

        # Parse the transaction text
        # Format: "TeamATraded • asset1 • asset2 to TeamB for • asset3 • asset4"
        # Can have multiple trades chained with dates

        # Split by "Traded" to find individual transactions
        # The team name precedes "Traded"
        parts = re.split(r"(\w[\w\s.'-]+?)Traded", transactions)

        # Process each trade mention
        i = 1
        while i < len(parts) - 1:
            from_team = parts[i].strip()
            trade_text = parts[i + 1] if i + 1 < len(parts) else ""

            # Split on " for • " or " for " to get the two sides
            sides = re.split(r"\s+for\s+•?\s*", trade_text, maxsplit=1)

            if len(sides) >= 2:
                sent_raw = sides[0].strip()
                received_raw = sides[1].strip()

                # Extract "to TeamB" from sent side
                to_match = re.search(r"\s+to\s+([\w\s.'-]+?)$", sent_raw)
                to_team = to_match.group(1).strip() if to_match else "Unknown"
                if to_match:
                    sent_raw = sent_raw[:to_match.start()]

                # Clean up assets: split on " • "
                def clean_assets(raw):
                    assets = re.split(r"\s*•\s*", raw)
                    cleaned = []
                    for a in assets:
                        a = a.strip()
                        # Remove pick result annotations like "(#14-Konsta Helenius)"
                        a = re.sub(r"\s*\([#\d]+-[^)]+\)", "", a)
                        # Remove "protected top X" notes
                        a = re.sub(r"\s*\(protected[^)]*\)", "", a)
                        # Remove "conditional" notes
                        a = re.sub(r"\s*\(conditional[^)]*\)", "", a)
                        # Remove trailing pick result
                        a = re.sub(r"\s*\(\?-\?\)", "", a)
                        a = a.strip()
                        if a and a != "cash" and a != "future considerations":
                            cleaned.append(a)
                    return cleaned

                sent = clean_assets(sent_raw)
                received = clean_assets(received_raw)

                if sent or received:
                    trade_id += 1
                    all_trades.append({
                        "trade_id": trade_id,
                        "draft_year": draft_year,
                        "team_1": from_team,
                        "team_2": to_team,
                        "team_1_sends": ";".join(sent),
                        "team_2_sends": ";".join(received),
                        "raw_text": transactions[:200],
                    })

            i += 2

print(f"\n{'='*60}")
print(f"Total trades parsed: {len(all_trades)}")

if all_trades:
    df = pd.DataFrame(all_trades)
    print(f"\nSample trades:")
    for _, t in df.head(20).iterrows():
        print(f"\n  Trade {t['trade_id']} ({t['draft_year']}): {t['team_1']} ↔ {t['team_2']}")
        print(f"    {t['team_1']} sends: {t['team_1_sends']}")
        print(f"    {t['team_2']} sends: {t['team_2_sends']}")

    out_path = os.path.join(BASE, "data", "trades.csv")
    df.drop(columns=["raw_text"]).to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} trades to {out_path}")
