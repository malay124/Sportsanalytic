"""Download regular-season box scores and closing lines from ESPN for backtesting.

    python research/build_dataset.py OUT_DIR

Writes OUT_DIR/games.csv (one row per game) and OUT_DIR/players.csv (one row per player-game).
Resumable: summaries are cached as OUT_DIR/raw/<event>.json.
"""
import csv
import datetime
import gzip
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SEASONS = {  # season label -> regular-season date range
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-10-21", "2026-04-12"),
}
STAT_KEYS = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return json.loads(data)
        except Exception:
            if i == tries - 1:
                raise


def event_ids(season, start, end):
    d, stop, out = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end), []
    days = []
    while d <= stop:
        days.append(d)
        d += datetime.timedelta(days=1)
    with ThreadPoolExecutor(12) as ex:
        for day, sb in zip(days, ex.map(lambda x: get(f"{SITE}/scoreboard?dates={x:%Y%m%d}"), days)):
            for e in sb.get("events", []):
                if e["season"]["type"] == 2 and e["status"]["type"]["completed"]:
                    out.append((season, day.isoformat(), e["id"]))
    return out


def summary(out_dir, eid):
    path = os.path.join(out_dir, "raw", f"{eid}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    s = get(f"{SITE}/summary?event={eid}")
    keep = {k: s.get(k) for k in ("boxscore", "header", "pickcenter")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f)
    return keep


def split_made(v):
    m, a = (v.split("-") + ["0"])[:2]
    return int(m or 0), int(a or 0)


def parse(season, date, eid, s):
    comp = s["header"]["competitions"][0]
    side = {c["homeAway"]: c for c in comp["competitors"]}
    pc = (s.get("pickcenter") or [{}])[0]
    game = {
        "game_id": eid, "season": season, "date": date, "neutral": int(bool(comp.get("neutralSite"))),
        "home": side["home"]["team"]["abbreviation"], "away": side["away"]["team"]["abbreviation"],
        "home_pts": int(side["home"]["score"]), "away_pts": int(side["away"]["score"]),
        "spread": pc.get("spread"), "total": pc.get("overUnder"),
        "home_ml": (pc.get("homeTeamOdds") or {}).get("moneyLine"),
        "away_ml": (pc.get("awayTeamOdds") or {}).get("moneyLine"),
    }
    rows = []
    for team in s["boxscore"]["players"]:
        abbr = team["team"]["abbreviation"]
        st = team["statistics"][0]
        idx = {k: st["labels"].index(k) for k in STAT_KEYS if k in st["labels"]}
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            v = a["stats"]
            fgm, fga = split_made(v[idx["FG"]])
            tpm, tpa = split_made(v[idx["3PT"]])
            ftm, fta = split_made(v[idx["FT"]])
            num = lambda k: int(v[idx[k]]) if v[idx[k]] not in ("", "--") else 0
            mins = num("MIN")
            if mins == 0:
                continue
            rows.append({
                "game_id": eid, "team": abbr, "opp": game["away"] if abbr == game["home"] else game["home"],
                "athlete_id": a["athlete"]["id"], "name": a["athlete"]["displayName"],
                "pos": (a["athlete"].get("position") or {}).get("abbreviation", ""),
                "starter": int(bool(a.get("starter"))), "min": mins, "pts": num("PTS"),
                "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                "reb": num("REB"), "oreb": num("OREB"), "dreb": num("DREB"), "ast": num("AST"),
                "tov": num("TO"), "stl": num("STL"), "blk": num("BLK"), "pf": num("PF"),
                "pm": int(v[idx["+/-"]].replace("+", "")) if v[idx["+/-"]] not in ("", "--") else 0,
            })
    return game, rows


CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
BOOK_PREFERENCE = ("DraftKings", "ESPN BET", "Caesars", "BetMGM", "FanDuel", "Betfair")


def core_line(eid):
    """Pre-game spread/total/moneylines from the core odds feed, for games whose summary has none."""
    try:
        items = [i for i in get(f"{CORE}/events/{eid}/competitions/{eid}/odds").get("items", [])
                 if "live" not in i["provider"]["name"].lower() and i.get("spread") is not None]
    except Exception:
        return None
    if not items:
        return None
    rank = lambda i: next((n for n, b in enumerate(BOOK_PREFERENCE) if b.lower() in i["provider"]["name"].lower()), 99)
    i = min(items, key=rank)
    return {"spread": i.get("spread"), "total": i.get("overUnder"),
            "home_ml": (i.get("homeTeamOdds") or {}).get("moneyLine"),
            "away_ml": (i.get("awayTeamOdds") or {}).get("moneyLine"), "book": i["provider"]["name"]}


def backfill_lines(games):
    missing = [g for g in games if g["spread"] is None]
    with ThreadPoolExecutor(16) as ex:
        for g, line in zip(missing, ex.map(lambda g: core_line(g["game_id"]), missing)):
            if line:
                g.update(line)
    print(f"backfilled lines for {sum(1 for g in missing if g['spread'] is not None)}/{len(missing)} games")


def main(out_dir):
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    events = []
    for season, (a, b) in SEASONS.items():
        ev = event_ids(season, a, b)
        print(f"{season}: {len(ev)} games", flush=True)
        events += ev
    games, players, failed = [], [], 0
    with ThreadPoolExecutor(16) as ex:
        results = ex.map(lambda e: (e, summary(out_dir, e[2])), events)
        for i, ((season, date, eid), s) in enumerate(results):
            try:
                g, rows = parse(season, date, eid, s)
                games.append(g)
                players += rows
            except Exception as err:
                failed += 1
                print(f"skip {eid}: {err}", flush=True)
            if i % 500 == 0:
                print(f"{i}/{len(events)}", flush=True)
    for g in games:
        g["book"] = "DraftKings" if g["spread"] is not None else ""
    backfill_lines(games)
    for name, data in (("games.csv", games), ("players.csv", players)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"done: {len(games)} games, {len(players)} player rows, {failed} failed")


if __name__ == "__main__":
    main(sys.argv[1])
