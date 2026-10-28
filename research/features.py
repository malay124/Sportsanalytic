"""Pre-game features for every team-game and player-game. Nothing uses information from the game itself
or later games: every rolling stat is computed from earlier games only (shifted expanding sums).

    python research/features.py DATA_DIR   ->  DATA_DIR/team_feats.csv, DATA_DIR/player_feats.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nba_picks.calibrate import shrink  # noqa: E402

K_TEAM, K_PLAYER, K_DVP = 10, 3, 30   # K_PLAYER: best out-of-sample k (fit_models.py)
POS = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}


def prior_season(s):
    a = int(s[:4]) - 1
    return f"{a}-{str(a + 1)[2:]}"


def pregame_mean(df, keys, num, den=None):
    """Mean of num (or ratio sum(num)/sum(den)) over earlier rows of the same group. Returns (value, n)."""
    g = df.groupby(keys, sort=False)
    n = g.cumcount()
    s_num = g[num].cumsum() - df[num]
    if den is None:
        return (s_num / n.replace(0, np.nan)), n
    s_den = g[den].cumsum() - df[den]
    return s_num / s_den.replace(0, np.nan), n


# ---------------------------------------------------------------- team-games

def team_games(games, players):
    p = players.copy()
    p["used"] = p.fga + 0.44 * p.fta + p.tov
    p["bench"] = 1 - p.starter
    for c in ("min", "pts", "pm", "fga", "fta", "used"):
        p[f"b_{c}"] = p[c] * p.bench
    p["rot"] = (p["min"] >= 12).astype(int)
    agg = p.groupby(["game_id", "team"]).agg(
        pts=("pts", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"),
        tmin=("min", "sum"), used=("used", "sum"), b_min=("b_min", "sum"), b_pts=("b_pts", "sum"),
        b_pm=("b_pm", "sum"), b_fga=("b_fga", "sum"), b_fta=("b_fta", "sum"), rot=("rot", "sum")).reset_index()
    agg["poss"] = agg.fga - agg.oreb + agg.tov + 0.44 * agg.fta

    rows = []
    for _, g in games.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            rows.append({"game_id": g.game_id, "season": g.season, "date": g.date, "team": g[side], "opp": g[opp],
                         "is_home": int(side == "home"), "neutral": g.neutral,
                         "pf": g[f"{side}_pts"], "pa": g[f"{opp}_pts"]})
    t = pd.DataFrame(rows).merge(agg.drop(columns="pts"), on=["game_id", "team"], how="left")
    opp_poss = agg[["game_id", "team", "poss"]].rename(columns={"team": "opp", "poss": "opp_poss"})
    t = t.merge(opp_poss, on=["game_id", "opp"], how="left")
    t["gposs"] = (t.poss + t.opp_poss) / 2
    t["net100"] = 100 * (t.pf - t.pa) / t.gposs
    t["ortg"] = 100 * t.pf / t.gposs
    t["drtg"] = 100 * t.pa / t.gposs
    t["b_net48"] = 48 * t.b_pm / t.b_min.replace(0, np.nan)
    t["b_min_share"] = t.b_min / t.tmin
    t["b_shot_den"] = 2 * (t.b_fga + 0.44 * t.b_fta)
    t = t.sort_values(["season", "team", "date", "game_id"]).reset_index(drop=True)

    # rest / schedule density
    d = pd.to_datetime(t.date)
    t["_d"] = d
    grp = t.groupby(["season", "team"], sort=False)["_d"]
    prev1, prev2, prev4 = grp.shift(1), grp.shift(2), grp.shift(4)
    t["rest_days"] = ((d - prev1).dt.days - 1).fillna(3).clip(upper=3)
    t["b2b"] = (t.rest_days == 0).astype(int)
    t["three_in_four"] = ((d - prev2).dt.days <= 3).astype(int)
    t["five_in_seven"] = ((d - prev4).dt.days <= 6).astype(int)
    t["road_b2b"] = t.b2b * (1 - t.is_home)

    # pre-game rolling ratings, shrunk toward last season's final value (league average for the first season)
    keys = ["season", "team"]
    for col, num, den in (("net", "net100", None), ("ortg_r", "pf", "gposs"), ("drtg_r", "pa", "gposs"),
                          ("pace", "gposs", None), ("bench_net", "b_pm", "b_min"), ("bench_share", "b_min", "tmin"),
                          ("bench_ts", "b_pts", "b_shot_den"), ("depth", "rot", None)):
        t[f"{col}_raw"], t["gp"] = pregame_mean(t, keys, num, den)
    t.ortg_r_raw *= 100
    t.drtg_r_raw *= 100
    t.bench_net_raw *= 48
    final = t.groupby(keys)[[c for c in t.columns if c.endswith("_raw")]].last()  # end-of-season (all-but-last game)
    lg = t[[c for c in t.columns if c.endswith("_raw")]].mean()
    for col in [c[:-4] for c in t.columns if c.endswith("_raw")]:
        pri = [final[f"{col}_raw"].get((prior_season(s), tm), lg[f"{col}_raw"]) for s, tm in zip(t.season, t.team)]
        t[col] = [shrink(v if pd.notna(v) else p0, n, p0, K_TEAM) for v, n, p0 in zip(t[f"{col}_raw"], t.gp, pri)]
    return t


def srs_ratings(t, lam=8.0):
    """Schedule-adjusted ratings (SRS-style) as of each date: ridge fit of net100 = r_team - r_opp + hca on
    the season's earlier games, shrunk toward last season's final ratings. Returns {(season, date, team): r}."""
    out, prior = {}, {}
    home = t[t.is_home == 1]
    for season in sorted(t.season.unique()):
        hs = home[home.season == season].sort_values("date")
        teams = sorted(set(hs.team) | set(hs.opp))
        ix = {x: i for i, x in enumerate(teams)}
        p0 = np.array([prior.get(x, 0.0) for x in teams])
        last = None
        for date in sorted(hs.date.unique()):
            past = hs[hs.date < date]
            n = len(past)
            X = np.zeros((n + len(teams), len(teams) + 1))
            y = np.zeros(n + len(teams))
            if n:
                X[np.arange(n), past.team.map(ix).values] = 1
                X[np.arange(n), past.opp.map(ix).values] = -1
                X[np.arange(n), -1] = 1 - past.neutral.values
                y[:n] = past.net100.values
            X[n + np.arange(len(teams)), np.arange(len(teams))] = np.sqrt(lam)
            y[n:] = np.sqrt(lam) * p0
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            r = coef[:-1] - (coef[:-1] - p0).mean() * 0  # keep level from prior
            for x in teams:
                out[(season, date, x)] = r[ix[x]]
            last = r
        prior = {x: last[ix[x]] for x in teams} if last is not None else prior
    return out


# ---------------------------------------------------------------- player-games

def player_games(games, players, t):
    p = players.merge(games[["game_id", "season", "date"]], on="game_id")
    p["posb"] = p.pos.map(POS).fillna("F")
    p["used"] = p.fga + 0.44 * p.fta + p.tov
    # only pre-game columns here: drtg/ortg are this game's actual values, drtg_r/pace are pre-game rolling
    tm = t[["game_id", "team", "used", "tmin", "b2b", "is_home", "gp"]].rename(
        columns={"used": "team_used", "tmin": "team_min", "gp": "team_gp"})
    p = p.merge(tm, on=["game_id", "team"], how="left")
    opp = t[["game_id", "team", "drtg_r", "pace"]].rename(columns={"team": "opp", "drtg_r": "opp_drtg",
                                                                    "pace": "opp_pace"})
    p = p.merge(opp, on=["game_id", "opp"], how="left")
    p["team_used_on"] = p.team_used * p["min"] / (p.team_min / 5)   # team possessions used while on court
    p["shot_den"] = 2 * (p.fga + 0.44 * p.fta)
    p = p.sort_values(["season", "athlete_id", "date", "game_id"]).reset_index(drop=True)

    keys = ["season", "athlete_id"]
    p["ppg_raw"], p["pgp"] = pregame_mean(p, keys, "pts")
    p["mpg"], _ = pregame_mean(p, keys, "min")
    p["usg"], _ = pregame_mean(p, keys, "used", "team_used_on")
    p["ppu"], _ = pregame_mean(p, keys, "pts", "used")             # points per used possession
    p["ts"], _ = pregame_mean(p, keys, "pts", "shot_den")
    p["ppm"], _ = pregame_mean(p, keys, "pts", "min")
    p["rpm"], _ = pregame_mean(p, keys, "reb", "min")
    p["apm"], _ = pregame_mean(p, keys, "ast", "min")

    # baseline points: season ppg shrunk toward last season's ppg (the model's current approach)
    last = p.groupby(keys).apply(lambda g: g.pts.mean(), include_groups=False)
    pri = [last.get((prior_season(s), a), np.nan) for s, a in zip(p.season, p.athlete_id)]
    p["prior_ppg"] = pri
    p["base"] = [shrink(v, n, q, K_PLAYER) if pd.notna(q) else v for v, n, q in zip(p.ppg_raw, p.pgp, pri)]

    # vacated usage: regulars missing from this game (played >= 20 mpg, appeared in the team's last 10 games)
    p["team_game_no"] = p.team_gp
    reg = p[p.mpg >= 20][["season", "team", "athlete_id", "team_game_no", "date", "usg", "mpg", "ppg_raw"]]
    tg = t[["season", "team", "game_id", "gp", "date"]].rename(columns={"gp": "team_game_no"})
    played = set(zip(p.game_id, p.athlete_id))
    # every appearance, to tell a traded player (now on another team) from an injured one
    appear = p.groupby("athlete_id")[["date", "team"]].apply(lambda g: list(zip(g.date, g.team))).to_dict()
    vac = {}
    for (season, team), grp in reg.groupby(["season", "team"]):
        games_t = tg[(tg.season == season) & (tg.team == team)]
        hist = grp.sort_values("team_game_no")
        for gm in games_t.itertuples():
            seen = hist[(hist.team_game_no < gm.team_game_no) & (hist.team_game_no >= gm.team_game_no - 10)]
            seen = seen.groupby("athlete_id").last()
            mask = np.array([(gm.game_id, a) not in played and
                             not any(tm2 != team and r.date < d2 <= gm.date for d2, tm2 in appear[a])
                             for a, r in seen.iterrows()], dtype=bool)
            miss = seen.loc[mask] if len(seen) else seen
            vac[gm.game_id, team] = (miss.usg.fillna(0).sum(), miss.mpg.sum(), miss.ppg_raw.fillna(0).sum()) \
                if len(miss) else (0.0, 0.0, 0.0)
    v = pd.Series(vac)
    p["vac_usg"] = [v.get((g, tmn), (0, 0, 0))[0] for g, tmn in zip(p.game_id, p.team)]
    p["vac_min"] = [v.get((g, tmn), (0, 0, 0))[1] for g, tmn in zip(p.game_id, p.team)]
    p["vac_ppg"] = [v.get((g, tmn), (0, 0, 0))[2] for g, tmn in zip(p.game_id, p.team)]

    # defense vs position: how far above their own baseline players of this position score against this opponent,
    # from earlier games only, shrunk toward 0 (k counted in player-games)
    p["resid"] = p.pts - p.base
    og = (p[p.pgp >= 5].groupby(["season", "opp", "posb", "date", "game_id"])
          .resid.agg(["sum", "count"]).reset_index().sort_values(["season", "opp", "posb", "date"]))
    g = og.groupby(["season", "opp", "posb"])
    og["s"] = g["sum"].cumsum() - og["sum"]
    og["c"] = g["count"].cumsum() - og["count"]
    og["dvp"] = og.s / (og.c + K_DVP)
    p = p.merge(og[["game_id", "opp", "posb", "dvp", "c"]].rename(columns={"c": "dvp_n"}),
                on=["game_id", "opp", "posb"], how="left")
    p["dvp"] = p.dvp.fillna(0.0)
    # how many teammates share the position bucket (a lone C sees the opposing C more of the time)
    return p


def main(data_dir):
    games = pd.read_csv(os.path.join(data_dir, "games.csv"))
    players = pd.read_csv(os.path.join(data_dir, "players.csv"))
    t = team_games(games, players)
    srs = srs_ratings(t)
    t["srs"] = [srs.get((s, d, x), np.nan) for s, d, x in zip(t.season, t.date, t.team)]
    # strength of schedule faced so far: mean current SRS of past opponents
    t["opp_srs_now"] = [srs.get((s, d, o), np.nan) for s, d, o in zip(t.season, t.date, t.opp)]
    t["sos"], _ = pregame_mean(t, ["season", "team"], "opp_srs_now")
    t["sos"] = t.sos.fillna(0.0)
    p = player_games(games, players, t)
    # team availability from the player table: share of regular minutes and points missing
    av = p.groupby(["game_id", "team"])[["vac_min", "vac_ppg", "vac_usg"]].first().reset_index()
    t = t.merge(av, on=["game_id", "team"], how="left")
    t.to_csv(os.path.join(data_dir, "team_feats.csv"), index=False)
    p.to_csv(os.path.join(data_dir, "player_feats.csv"), index=False)
    print(f"team rows {len(t)}, player rows {len(p)}")


if __name__ == "__main__":
    main(sys.argv[1])
