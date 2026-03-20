"""
scrape_nhl.py — Scrape NHL play-by-play + shifts for the current season.

Pulls from the NHL API:
  - PBP events:  https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play
  - Shift charts: https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}

Outputs match the format of existing data files:
  - data/raw_pbp_2025.csv      (PBP events in raw_pbp.csv format)
  - data/raw_data_2025.csv     (lineup CHANGE events in raw_data.csv format)

Usage:
  python scraper/scrape_nhl.py                    # scrape full 2025-26 season
  python scraper/scrape_nhl.py --season 2024      # scrape 2024-25 season
  python scraper/scrape_nhl.py --start 2025-12-01 # scrape from a specific date
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
SHIFTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}"
SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"

REQUEST_DELAY = 0.35  # seconds between API calls (be polite)
MAX_RETRIES = 3

# Event type mapping: NHL API typeDescKey → our event_type
EVENT_TYPE_MAP = {
    "faceoff": "FACEOFF",
    "hit": "HIT",
    "shot-on-goal": "SHOT",
    "goal": "GOAL",
    "missed-shot": "MISSED_SHOT",
    "blocked-shot": "BLOCKED_SHOT",
    "giveaway": "GIVEAWAY",
    "takeaway": "TAKEAWAY",
    "penalty": "PENALTY",
    "stoppage": "STOPPAGE",
    "period-start": "PERIOD_START",
    "period-end": "PERIOD_END",
    "game-end": "GAME_END",
    "delayed-penalty": "DELAYED_PENALTY",
    "failed-shot-attempt": "FAILED_SHOT",
    "shootout-complete": "SHOOTOUT_COMPLETE",
}

# situationCode format: 4 digits → [away_goalie, away_skaters, home_skaters, home_goalie]
# e.g., 1551 = away: 1 goalie + 5 skaters, home: 5 skaters + 1 goalie


# ── HTTP helpers ─────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({"User-Agent": "nhl-rapm-scraper/1.0"})


def fetch_json(url, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                print(f"  HTTP {resp.status_code} for {url}, retry {attempt+1}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"  Request error: {e}, retry {attempt+1}", file=sys.stderr)
        time.sleep(2 ** attempt)
    return None


# ── Schedule ─────────────────────────────────────────────────────────────────

def get_season_games(season_start_year, start_date=None):
    """Get all regular season game IDs for a season by walking the schedule."""
    # NHL regular season: early October → mid April
    if start_date:
        current = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        current = datetime(season_start_year, 10, 1)

    end = datetime(season_start_year + 1, 4, 30)
    today = datetime.now()
    if end > today:
        end = today - timedelta(days=1)

    games = []
    seen_ids = set()

    print(f"Fetching schedule from {current.date()} to {end.date()}...", file=sys.stderr)

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        data = fetch_json(SCHEDULE_URL.format(date=date_str))
        time.sleep(REQUEST_DELAY)

        if data and "gameWeek" in data:
            for week in data["gameWeek"]:
                for game in week.get("games", []):
                    gid = game["id"]
                    gtype = game.get("gameType", 0)
                    gstate = game.get("gameState", "")

                    # Only regular season (type 2), completed games
                    if gtype == 2 and gstate in ("OFF", "FINAL") and gid not in seen_ids:
                        games.append({
                            "game_id": gid,
                            "date": week["date"],
                            "home_id": game["homeTeam"]["id"],
                            "home_abbrev": game["homeTeam"]["abbrev"],
                            "away_id": game["awayTeam"]["id"],
                            "away_abbrev": game["awayTeam"]["abbrev"],
                            "home_score": game["homeTeam"].get("score", 0),
                            "away_score": game["awayTeam"].get("score", 0),
                        })
                        seen_ids.add(gid)

            # Jump forward by the length of the gameWeek response
            dates_in_response = [week["date"] for week in data["gameWeek"]]
            if dates_in_response:
                last_date = max(dates_in_response)
                current = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            else:
                current += timedelta(days=7)
        else:
            current += timedelta(days=7)

    games.sort(key=lambda g: (g["date"], g["game_id"]))
    print(f"Found {len(games)} regular season games", file=sys.stderr)
    return games


# ── HTML shift report fallback ────────────────────────────────────────────────

import re


def fetch_html_shifts(game_id, season_start_year, roster_spots):
    """
    Fallback: parse NHL HTML shift reports when the API returns empty.
    Uses rosterSpots from the PBP response to map jersey numbers → player IDs.
    """
    season_str = f"{season_start_year}{season_start_year + 1}"
    game_num = game_id % 100000

    # Build jersey → player_id lookup from rosterSpots, keyed by (team_id, jersey)
    jersey_to_pid = {}
    pid_to_info = {}
    for rs in roster_spots:
        tid = rs["teamId"]
        jersey = rs.get("sweaterNumber", 0)
        pid = rs["playerId"]
        first = rs.get("firstName", {}).get("default", "")
        last = rs.get("lastName", {}).get("default", "")
        jersey_to_pid[(tid, jersey)] = pid
        pid_to_info[pid] = {"firstName": first, "lastName": last, "teamId": tid}

    shifts = []

    for team_type, prefix in [("home", "TH"), ("away", "TV")]:
        url = f"https://www.nhl.com/scores/htmlreports/{season_str}/{prefix}0{game_num:05d}.HTM"
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            continue

        content = resp.text

        # Extract team name from heading
        team_match = re.search(r'teamHeading[^>]*>([^<]+)', content)
        team_name = team_match.group(1).strip() if team_match else ""

        # Parse player sections
        player_sections = re.split(r'class="playerHeading', content)

        for section in player_sections[1:]:
            name_match = re.search(r'>\s*(\d+)\s+([A-Z][A-Z\s\',.-]+)', section)
            if not name_match:
                continue

            jersey = int(name_match.group(1))
            raw_name = name_match.group(2).strip()
            parts = raw_name.split(',')
            if len(parts) == 2:
                first = parts[1].strip().title()
                last = parts[0].strip().title()
            else:
                first = ""
                last = raw_name.title()

            # Map jersey → player_id via rosterSpots
            player_id = 0
            for (tid, j), pid in jersey_to_pid.items():
                if j == jersey:
                    info = pid_to_info[pid]
                    # Verify by name similarity
                    if (info["lastName"].lower().startswith(last[:3].lower()) or
                            last.lower().startswith(info["lastName"][:3].lower())):
                        player_id = pid
                        first = info["firstName"]
                        last = info["lastName"]
                        break
            # Fallback: just match by jersey if only one match
            if player_id == 0:
                matches = [(tid, pid) for (tid, j), pid in jersey_to_pid.items() if j == jersey]
                if len(matches) == 1:
                    player_id = matches[0][1]
                    info = pid_to_info[player_id]
                    first = info["firstName"]
                    last = info["lastName"]

            # Parse shift rows
            shift_pattern = re.findall(
                r'<td[^>]*>\s*(\d+)\s*</td>\s*'
                r'<td[^>]*>\s*(\d+)\s*</td>\s*'
                r'<td[^>]*>\s*(\d+:\d+)\s*/\s*\d+:\d+\s*</td>\s*'
                r'<td[^>]*>\s*(\d+:\d+)\s*/\s*\d+:\d+\s*</td>\s*'
                r'<td[^>]*>\s*(\d+:\d+)\s*</td>',
                section
            )

            for shift_num, period, start_time, end_time, duration in shift_pattern:
                shifts.append({
                    "playerId": player_id,
                    "firstName": first,
                    "lastName": last,
                    "teamId": 0,
                    "teamAbbrev": "",
                    "teamName": team_name,
                    "period": int(period),
                    "startTime": start_time,
                    "endTime": end_time,
                    "duration": duration,
                    "shiftNumber": int(shift_num),
                    "typeCode": 517,
                    "_team_type": team_type,
                })

    return shifts


# ── Shift parsing → on-ice tracking ─────────────────────────────────────────

def parse_time_to_seconds(time_str):
    """Convert MM:SS to seconds elapsed in period."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def build_on_ice_from_shifts(shifts_data, home_id, away_id):
    """
    Build a lookup: (period, elapsed_second) → {home_skaters, away_skaters, home_goalie, away_goalie}

    Returns a function that, given a period and elapsed seconds, returns who's on ice.
    """
    if not shifts_data:
        return None

    # Parse shifts into intervals
    shifts = []
    for s in shifts_data:
        try:
            start_sec = parse_time_to_seconds(s["startTime"])
            end_sec = parse_time_to_seconds(s["endTime"]) if s["endTime"] else start_sec
            # Duration check — goalie shifts can span full period
            shifts.append({
                "player_id": s["playerId"],
                "first_name": s.get("firstName", ""),
                "last_name": s.get("lastName", ""),
                "team_id": s["teamId"],
                "period": s["period"],
                "start": start_sec,
                "end": max(end_sec, start_sec + 1),  # ensure non-zero duration
                "type_code": s.get("typeCode", 517),
            })
        except (ValueError, KeyError):
            continue

    def get_on_ice(period, elapsed_sec):
        """Get players on ice at a specific moment."""
        home_players = []
        away_players = []
        home_goalie = None
        away_goalie = None

        for s in shifts:
            if s["period"] != period:
                continue
            if s["start"] <= elapsed_sec < s["end"]:
                pid = s["player_id"]
                name = f"{s['first_name']}.{s['last_name']}".replace(" ", ".")

                if s["team_id"] == home_id:
                    home_players.append((pid, name))
                elif s["team_id"] == away_id:
                    away_players.append((pid, name))

        # Separate goalies (typically the player with the longest shift in a period)
        # Heuristic: goalies have very long shifts, but we'll identify them by
        # checking which player has the most total ice time
        # For now, return all players and let downstream handle goalie separation
        return home_players, away_players

    return get_on_ice, shifts


def identify_goalies(shifts_data, home_id, away_id):
    """Identify goalies by finding players with the most total ice time per team per period."""
    if not shifts_data:
        return set(), set()

    # Goalies typically have shift durations close to 20:00 per period
    player_toi = {}
    for s in shifts_data:
        try:
            start_sec = parse_time_to_seconds(s["startTime"])
            end_sec = parse_time_to_seconds(s["endTime"]) if s["endTime"] else start_sec
            duration = max(end_sec - start_sec, 0)
            key = (s["playerId"], s["teamId"])
            player_toi[key] = player_toi.get(key, 0) + duration
        except (ValueError, KeyError):
            continue

    # For each team, the player with the most total ice time is likely the goalie
    home_goalies = set()
    away_goalies = set()

    home_players = {k: v for k, v in player_toi.items() if k[1] == home_id}
    away_players = {k: v for k, v in player_toi.items() if k[1] == away_id}

    if home_players:
        # Goalies typically have >2400 sec (40+ min)
        for (pid, tid), toi in home_players.items():
            if toi > 2000:
                home_goalies.add(pid)
        # Fallback: player with max TOI
        if not home_goalies:
            max_pid = max(home_players, key=home_players.get)[0]
            home_goalies.add(max_pid)

    if away_players:
        for (pid, tid), toi in away_players.items():
            if toi > 2000:
                away_goalies.add(pid)
        if not away_goalies:
            max_pid = max(away_players, key=away_players.get)[0]
            away_goalies.add(max_pid)

    return home_goalies, away_goalies


# ── PBP parsing ──────────────────────────────────────────────────────────────

def parse_pbp_game(game_info, pbp_data, shifts_data):
    """Parse a single game's PBP + shifts into rows matching raw_pbp.csv format."""
    game_id = game_info["game_id"]
    season = int(str(game_id)[:4])
    home_id = game_info["home_id"]
    away_id = game_info["away_id"]

    # Build on-ice lookup from shifts
    goalie_lookup_fn = None
    home_goalies, away_goalies = set(), set()
    on_ice_fn = None

    if shifts_data and len(shifts_data) > 0:
        on_ice_fn, parsed_shifts = build_on_ice_from_shifts(shifts_data, home_id, away_id)
        home_goalies, away_goalies = identify_goalies(shifts_data, home_id, away_id)

    plays = pbp_data.get("plays", [])
    home_team_name = pbp_data.get("homeTeam", {}).get("commonName", {})
    away_team_name = pbp_data.get("awayTeam", {}).get("commonName", {})
    home_abbrev = pbp_data.get("homeTeam", {}).get("abbrev", game_info["home_abbrev"])
    away_abbrev = pbp_data.get("awayTeam", {}).get("abbrev", game_info["away_abbrev"])

    # Handle team name — can be dict or string in API
    if isinstance(home_team_name, dict):
        home_team_name = home_team_name.get("default", home_abbrev)
    if isinstance(away_team_name, dict):
        away_team_name = away_team_name.get("default", away_abbrev)

    rows = []
    event_idx = 0

    for play in plays:
        event_idx += 1
        details = play.get("details", {})
        period_desc = play.get("periodDescriptor", {})
        period = period_desc.get("number", 0)
        period_type = period_desc.get("periodType", "REG")

        # Time parsing
        time_in_period = play.get("timeInPeriod", "00:00")
        time_remaining = play.get("timeRemaining", "20:00")
        period_seconds = parse_time_to_seconds(time_in_period)
        period_seconds_remaining = parse_time_to_seconds(time_remaining)
        game_seconds = (period - 1) * 1200 + period_seconds

        # Event type
        type_key = play.get("typeDescKey", "unknown")
        event_type = EVENT_TYPE_MAP.get(type_key, type_key.upper().replace("-", "_"))

        # Situation code → strength state
        sit_code = str(play.get("situationCode", "1551"))
        if len(sit_code) == 4:
            away_goalie_on = int(sit_code[0])
            away_skaters = int(sit_code[1])
            home_skaters = int(sit_code[2])
            home_goalie_on = int(sit_code[3])
            strength_state = f"{away_skaters}v{home_skaters}"
        else:
            away_skaters, home_skaters = 5, 5
            away_goalie_on, home_goalie_on = 1, 1
            strength_state = "5v5"

        # Event team
        event_team_id = details.get("eventOwnerTeamId", None)
        if event_team_id == home_id:
            event_team = home_team_name
            event_team_type = "home"
            event_team_abbr = home_abbrev
        elif event_team_id == away_id:
            event_team = away_team_name
            event_team_type = "away"
            event_team_abbr = away_abbrev
        else:
            event_team = ""
            event_team_type = ""
            event_team_abbr = ""

        # Coordinates
        x = details.get("xCoord", "")
        y = details.get("yCoord", "")

        # Scores
        home_score = details.get("homeScore", play.get("homeScore", 0)) or 0
        away_score = details.get("awayScore", play.get("awayScore", 0)) or 0

        # Players
        ep1_id = (details.get("shootingPlayerId") or details.get("scoringPlayerId")
                  or details.get("hittingPlayerId") or details.get("committedByPlayerId")
                  or details.get("playerId") or details.get("winningPlayerId") or "")
        ep2_id = (details.get("goalieInNetId") or details.get("hitteePlayerId")
                  or details.get("drawnByPlayerId") or details.get("losingPlayerId") or "")
        ep3_id = details.get("assist1PlayerId", "")
        ep4_id = details.get("assist2PlayerId", "")

        # Goalie info
        goalie_in_net = details.get("goalieInNetId", "")

        # Shot info
        shot_type = details.get("shotType", "")

        # Penalty info
        penalty_severity = details.get("descKey", "")
        penalty_minutes = details.get("duration", "")

        # Build event_id in the same format: {game_id}{event_idx:04d}
        event_id = int(f"{game_id}{event_idx:04d}")

        # Empty net / game winning goal
        empty_net = ""
        gwg = ""
        if event_type == "GOAL":
            # Check if goalie was pulled (goalie_on = 0 for defending team)
            if event_team_type == "home" and away_goalie_on == 0:
                empty_net = True
            elif event_team_type == "away" and home_goalie_on == 0:
                empty_net = True

        row = {
            "game_id": game_id,
            "season": season + 1,  # Convert to end-year format (2025 season → 2026)
            "event_type": event_type,
            "event": type_key,
            "event_id": event_id,
            "description": "",
            "period": period,
            "period_seconds": period_seconds,
            "period_seconds_remaining": period_seconds_remaining,
            "game_seconds": game_seconds,
            "game_seconds_remaining": "",
            "home_score": home_score,
            "away_score": away_score,
            "strength_state": strength_state,
            "event_idx": f"{event_idx:04d}",
            "extra_attacker": "",
            "home_skaters": home_skaters,
            "away_skaters": away_skaters,
            "period_type": period_type,
            "home_final": game_info["home_score"],
            "away_final": game_info["away_score"],
            "season_type": "R",
            "game_date": game_info["date"],
            "game_start": "",
            "game_end": "",
            "game_state": "Final",
            "venue_id": "",
            "home_name": home_team_name,
            "home_abbreviation": home_abbrev,
            "home_id": home_id,
            "away_name": away_team_name,
            "away_abbreviation": away_abbrev,
            "away_id": away_id,
            "event_team": event_team,
            "event_team_type": event_team_type,
            "home_goalie": "",
            "away_goalie": "",
            "event_player_1_name": "",
            "event_player_1_type": "",
            "event_player_2_name": "",
            "event_player_2_type": "",
            "strength_code": "",
            "strength": "",
            "x": x,
            "y": y,
            "x_fixed": "",
            "y_fixed": "",
            "event_player_1_id": ep1_id,
            "event_player_1_link": "",
            "event_player_2_id": ep2_id,
            "event_player_2_link": "",
            "event_team_id": event_team_id or "",
            "event_team_link": "",
            "event_team_abbr": event_team_abbr,
            "shot_distance": "",
            "shot_angle": "",
            "num_off": "",
            "away_on_5": "",
            "event_goalie_name": "",
            "event_goalie_id": goalie_in_net,
            "event_goalie_link": "",
            "event_goalie_type": "",
            "event_player_3_name": "",
            "event_player_3_type": "",
            "game_winning_goal": gwg,
            "empty_net": empty_net,
            "event_player_3_id": ep3_id,
            "event_player_3_link": "",
            "event_player_4_type": "",
            "event_player_4_id": ep4_id,
            "event_player_4_name": "",
            "event_player_4_link": "",
            "penalty_severity": penalty_severity,
            "penalty_minutes": penalty_minutes,
        }
        rows.append(row)

    return rows


def build_lineup_events(game_info, shifts_data):
    """
    Build CHANGE-style lineup events from shift data.
    Outputs rows matching raw_data.csv format.

    Strategy: at each moment where the on-ice personnel changes (shift start/end),
    emit a CHANGE event with the current on-ice lineup.
    """
    if not shifts_data:
        return []

    game_id = game_info["game_id"]
    home_id = game_info["home_id"]
    away_id = game_info["away_id"]

    home_goalies, away_goalies = identify_goalies(shifts_data, home_id, away_id)

    # Parse all shift boundaries
    boundaries = set()
    parsed = []
    for s in shifts_data:
        try:
            start_sec = parse_time_to_seconds(s["startTime"])
            end_sec = parse_time_to_seconds(s["endTime"]) if s["endTime"] else start_sec
            period = s["period"]
            parsed.append({
                "player_id": s["playerId"],
                "name": f"{s.get('firstName', '')}.{s.get('lastName', '')}".replace(" ", "."),
                "team_id": s["teamId"],
                "period": period,
                "start": start_sec,
                "end": max(end_sec, start_sec + 1),
            })
            boundaries.add((period, start_sec))
            boundaries.add((period, end_sec))
        except (ValueError, KeyError):
            continue

    rows = []
    event_counter = 0

    for period, sec in sorted(boundaries):
        # Find everyone on ice at this moment
        home_on = []
        away_on = []
        home_goalie_name = ""
        away_goalie_name = ""
        home_ids = []
        away_ids = []

        for s in parsed:
            if s["period"] == period and s["start"] <= sec < s["end"]:
                pid = s["player_id"]
                name = s["name"]
                if s["team_id"] == home_id:
                    if pid in home_goalies:
                        home_goalie_name = name
                    else:
                        home_on.append(name)
                        home_ids.append(str(pid))
                elif s["team_id"] == away_id:
                    if pid in away_goalies:
                        away_goalie_name = name
                    else:
                        away_on.append(name)
                        away_ids.append(str(pid))

        if not home_on and not away_on:
            continue

        event_counter += 1
        event_id = int(f"{game_id}{event_counter:04d}")

        # Pad to 6 slots
        while len(home_on) < 6:
            home_on.append("")
        while len(away_on) < 6:
            away_on.append("")

        row = {
            "game_id": game_id,
            "event_id": event_id,
            "event_team": "",
            "event_team_type": "",
            "players_on": "",
            "players_off": "",
            "ids_on": ", ".join(home_ids + away_ids),
            "ids_off": "",
            "away_on_1": away_on[0] if len(away_on) > 0 else "",
            "away_on_2": away_on[1] if len(away_on) > 1 else "",
            "away_on_3": away_on[2] if len(away_on) > 2 else "",
            "away_on_4": away_on[3] if len(away_on) > 3 else "",
            "away_on_5": away_on[4] if len(away_on) > 4 else "",
            "away_on_6": away_on[5] if len(away_on) > 5 else "",
            "away_goalie": away_goalie_name,
            "home_on_1": home_on[0] if len(home_on) > 0 else "",
            "home_on_2": home_on[1] if len(home_on) > 1 else "",
            "home_on_3": home_on[2] if len(home_on) > 2 else "",
            "home_on_4": home_on[3] if len(home_on) > 3 else "",
            "home_on_5": home_on[4] if len(home_on) > 4 else "",
            "home_on_6": home_on[5] if len(home_on) > 5 else "",
            "home_goalie": home_goalie_name,
        }
        rows.append(row)

    return rows


# ── Shots data download ─────────────────────────────────────────────────────

def download_shots(season_start_year):
    """Download MoneyPuck shots file for a season."""
    import zipfile
    import io

    url = f"https://peter-tanner.com/moneypuck/downloads/shots_{season_start_year}.zip"
    print(f"Downloading shots from {url}...", file=sys.stderr)

    resp = session.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"  Failed to download shots: HTTP {resp.status_code}", file=sys.stderr)
        print(f"  Shots may not be available yet for {season_start_year}-{season_start_year+1}",
              file=sys.stderr)
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        print(f"  Extracting {csv_name}...", file=sys.stderr)
        outpath = os.path.join(DATA_DIR, f"shots_{season_start_year}.csv")
        with zf.open(csv_name) as src, open(outpath, "wb") as dst:
            dst.write(src.read())
        print(f"  Saved to {outpath}", file=sys.stderr)
        return outpath


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape NHL PBP + shifts data")
    parser.add_argument("--season", type=int, default=2025,
                        help="Season start year (default: 2025 for 2025-26)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD) to resume scraping from")
    parser.add_argument("--skip-shots", action="store_true",
                        help="Skip downloading MoneyPuck shots file")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Limit number of games to scrape (for testing)")
    args = parser.parse_args()

    season = args.season
    print(f"Scraping {season}-{season+1} NHL season", file=sys.stderr)

    # 1. Get schedule
    games = get_season_games(season, start_date=args.start)
    if args.max_games:
        games = games[:args.max_games]

    if not games:
        print("No games found!", file=sys.stderr)
        return

    # 2. Download shots from MoneyPuck
    if not args.skip_shots:
        download_shots(season)

    # 3. Scrape PBP + shifts for each game
    all_pbp_rows = []
    all_lineup_rows = []
    failed_games = []

    for i, game in enumerate(games):
        gid = game["game_id"]
        pct = (i + 1) / len(games) * 100
        print(f"\r  [{i+1}/{len(games)}] ({pct:.0f}%) Game {gid} ({game['date']}) "
              f"{game['away_abbrev']}@{game['home_abbrev']}...",
              end="", file=sys.stderr)

        # Fetch PBP
        pbp = fetch_json(PBP_URL.format(game_id=gid))
        time.sleep(REQUEST_DELAY)

        # Fetch shifts (try API first, fall back to HTML reports)
        shifts_resp = fetch_json(SHIFTS_URL.format(game_id=gid))
        shifts_data = shifts_resp.get("data", []) if shifts_resp else []
        time.sleep(REQUEST_DELAY)

        if not pbp:
            failed_games.append(gid)
            continue

        # If API shifts are empty, try HTML shift reports
        if not shifts_data and pbp:
            roster_spots = pbp.get("rosterSpots", [])
            if roster_spots:
                html_shifts = fetch_html_shifts(gid, season, roster_spots)
                if html_shifts:
                    # Convert HTML shifts to API format for downstream functions
                    # Need to assign teamId from the game info
                    for s in html_shifts:
                        if s.get("_team_type") == "home":
                            s["teamId"] = game["home_id"]
                        else:
                            s["teamId"] = game["away_id"]
                    shifts_data = html_shifts
                    time.sleep(REQUEST_DELAY)  # polite delay for HTML fetches

        # Parse PBP
        pbp_rows = parse_pbp_game(game, pbp, shifts_data)
        all_pbp_rows.extend(pbp_rows)

        # Parse lineups from shifts
        lineup_rows = build_lineup_events(game, shifts_data)
        all_lineup_rows.extend(lineup_rows)

    print(f"\n\nScraped {len(games) - len(failed_games)} games successfully", file=sys.stderr)
    if failed_games:
        print(f"Failed games ({len(failed_games)}): {failed_games[:10]}...", file=sys.stderr)

    # 4. Save outputs
    print("\nSaving outputs...", file=sys.stderr)

    pbp_df = pd.DataFrame(all_pbp_rows)
    pbp_path = os.path.join(DATA_DIR, f"raw_pbp_{season}.csv")
    pbp_df.to_csv(pbp_path, index=False)
    print(f"  {len(pbp_df):,} PBP rows → {pbp_path}", file=sys.stderr)

    lineup_df = pd.DataFrame(all_lineup_rows)
    lineup_path = os.path.join(DATA_DIR, f"raw_data_{season}.csv")
    lineup_df.to_csv(lineup_path, index=False)
    print(f"  {len(lineup_df):,} lineup rows → {lineup_path}", file=sys.stderr)

    # 5. Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Season {season}-{season+1} scrape complete!", file=sys.stderr)
    print(f"  PBP events: {len(pbp_df):,}", file=sys.stderr)
    print(f"  Lineup events: {len(lineup_df):,}", file=sys.stderr)
    print(f"  Games: {len(games) - len(failed_games)}", file=sys.stderr)
    print(f"\nOutputs:", file=sys.stderr)
    print(f"  {pbp_path}", file=sys.stderr)
    print(f"  {lineup_path}", file=sys.stderr)
    shots_path = os.path.join(DATA_DIR, f"shots_{season}.csv")
    if os.path.exists(shots_path):
        print(f"  {shots_path}", file=sys.stderr)
    print(f"\nTo integrate into the pipeline, append these to the main data files", file=sys.stderr)
    print(f"or update supporting/build_dataset.py to read the season-specific files.", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
