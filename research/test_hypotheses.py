"""Test each hypothesis out of sample. Train on 2023-24 + 2024-25, test on 2025-26 (never touched in fitting).

    python research/test_hypotheses.py DATA_DIR > research/results.txt

Team questions are asked two ways:
  A. Does the feature improve a model that predicts the margin on its own (no betting line)?
  B. Does it predict the margin *beyond the closing spread*? Only B can make money.
Player questions: does the feature improve a points prediction beyond the shrunk season average?
Significance: paired bootstrap of the test-set error difference, resampling whole dates (games on one
night share information, so rows are not independent). In-sample group F-tests use HC3 robust errors.
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN, TEST = ["2023-24", "2024-25"], ["2025-26"]
RNG = np.random.default_rng(7)
BREAKEVEN = 110 / 210


def boot(err_a, err_b, dates, reps=2000):
    """Bootstrap the mean of (err_b - err_a) over dates. Negative = b is better. Returns (mean, lo, hi, p)."""
    df = pd.DataFrame({"d": dates, "x": err_b - err_a}).groupby("d").x.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = RNG.integers(0, len(df), (reps, len(df)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    m = s.sum() / c.sum()
    p = 2 * min((bs >= 0).mean(), (bs <= 0).mean())
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / reps)


def fit_ols(train, test, cols, target):
    X = sm.add_constant(train[cols], has_constant="add")
    m = sm.OLS(train[target], X).fit(cov_type="HC3")
    pred = m.predict(sm.add_constant(test[cols], has_constant="add"))
    return m, pred


# ---------------------------------------------------------------- team

def team_table(data_dir):
    t = pd.read_csv(os.path.join(data_dir, "team_feats.csv"))
    g = pd.read_csv(os.path.join(data_dir, "games.csv"))
    p = pd.read_csv(os.path.join(data_dir, "player_feats.csv"))
    t["rest_days"] = t.rest_days.clip(0, 3)
    # usage/availability-based team scoring projection from the players who suited up:
    # each player's per-minute scoring times his season minutes, rescaled to 240 team minutes
    # rotation players only: who else appears (garbage-time bench) depends on the game's outcome
    q = p[(p.pgp >= 3) & (p.mpg >= 12)].copy()
    q["xp"] = q.ppm * q.mpg
    agg = q.groupby(["game_id", "team"]).agg(xp=("xp", "sum"), xm=("mpg", "sum")).reset_index()
    agg["usage_proj"] = agg.xp * 240 / agg.xm
    t = t.merge(agg[["game_id", "team", "usage_proj"]], on=["game_id", "team"], how="left")
    t["team_proj"] = t.ortg_r * t.pace / 100
    t["usage_edge"] = t.usage_proj - t.team_proj        # how much the active roster differs from the season team
    cols = ["net", "srs", "sos", "rest_days", "b2b", "three_in_four", "five_in_seven", "road_b2b", "bench_net",
            "bench_ts", "bench_share", "depth", "vac_min", "vac_ppg", "usage_edge", "gp", "team_proj", "usage_proj",
            "pf"]
    h = t[t.is_home == 1][["game_id"] + cols].add_prefix("h_").rename(columns={"h_game_id": "game_id"})
    a = t[t.is_home == 0][["game_id"] + cols].add_prefix("a_").rename(columns={"a_game_id": "game_id"})
    d = g.merge(h, on="game_id").merge(a, on="game_id")
    for c in cols:
        d[f"d_{c}"] = d[f"h_{c}"] - d[f"a_{c}"]
    d["hca"] = 1 - d.neutral
    d["margin"] = d.home_pts - d.away_pts
    d["ats"] = d.margin + d.spread          # > 0: home covered
    return d.dropna(subset=["spread", "d_net", "d_srs", "d_usage_edge"]), t


GROUPS = {
    "base: net rating + home court": ["hca", "d_net"],
    "SOS: schedule-adjusted rating (SRS) instead of raw net": ["hca", "d_srs"],
    "SOS: + strength of schedule faced": ["hca", "d_net", "d_sos"],
    "rest: + rest days, B2B, 3-in-4, 5-in-7": ["hca", "d_net", "d_rest_days", "d_b2b", "d_three_in_four",
                                               "d_five_in_seven"],
    "bench: + bench net, bench TS%, bench minutes share, depth": ["hca", "d_net", "d_bench_net", "d_bench_ts",
                                                                 "d_bench_share", "d_depth"],
    "usage: + active-roster usage projection, missing regulars": ["hca", "d_net", "d_usage_edge", "d_vac_ppg"],
}
ALL = sorted({c for v in GROUPS.values() for c in v} - {"d_net"}) + ["d_net"]


def team_tests(d, out):
    tr, te = d[d.season.isin(TRAIN)], d[d.season.isin(TEST)]
    out(f"TEAM GAMES  train {len(tr)}  test {len(te)}\n")

    out("A. Predicting the home margin without the betting line (test-set RMSE, lower is better)")
    base_m, base_p = fit_ols(tr, te, GROUPS["base: net rating + home court"], "margin")
    base_e = (te.margin - base_p) ** 2
    sig = np.sqrt(base_m.mse_resid)
    for name, cols in list(GROUPS.items()) + [("everything", ALL)]:
        m, pr = fit_ols(tr, te, cols, "margin")
        e = (te.margin - pr) ** 2
        ll = -np.mean(np.log(np.where(te.margin > 0, stats.norm.cdf(pr / sig), 1 - stats.norm.cdf(pr / sig))))
        mean, lo, hi, p = boot(base_e, e, te.date)
        out(f"  {name:<62} RMSE {np.sqrt(e.mean()):6.3f}  win logloss {ll:.4f}  "
            f"dMSE vs base {mean:+6.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
    mk = (te.margin + te.spread) ** 2
    ll = -np.mean(np.log(np.where(te.margin > 0, stats.norm.cdf(-te.spread / sig), 1 - stats.norm.cdf(-te.spread / sig))))
    out(f"  {'(reference) closing spread alone':<62} RMSE {np.sqrt(mk.mean()):6.3f}  win logloss {ll:.4f}")

    out("\n   coefficients (all three seasons, HC3 robust), margin model with everything:")
    full = sm.OLS(d.margin, sm.add_constant(d[ALL])).fit(cov_type="HC3")
    for c in ALL:
        out(f"     {c:<20} {full.params[c]:+8.3f}  se {full.bse[c]:.3f}  p={full.pvalues[c]:.4f}")

    out("\nB. Beyond the closing spread: target = margin + spread (ATS result). Zero means the line was right.")
    out(f"  ATS residual sd {d.ats.std():.2f}; mean {d.ats.mean():+.2f}")
    full = sm.OLS(d.ats, sm.add_constant(d[ALL])).fit(cov_type="HC3")
    for name, cols in GROUPS.items():
        extra = [c for c in cols if c not in ("hca",)]
        r = np.zeros((len(extra), len(full.params)))
        for i, c in enumerate(extra):
            r[i, list(full.params.index).index(c)] = 1
        f = full.wald_test(r, scalar=True)
        out(f"  {name:<62} joint Wald p={float(f.pvalue):.3f}")
    out("   coefficients:")
    for c in ALL:
        out(f"     {c:<20} {full.params[c]:+8.3f}  se {full.bse[c]:.3f}  p={full.pvalues[c]:.4f}")
    m, pr = fit_ols(tr, te, ALL, "ats")
    zero = te.ats ** 2
    e = (te.ats - pr) ** 2
    mean, lo, hi, p = boot(zero, e, te.date)
    out(f"  out of sample: MSE change vs 'line is right' {mean:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")
    for thr in (0.0, 1.0, 2.0):
        pick = te[np.abs(pr) > thr]
        side = np.sign(pr[np.abs(pr) > thr])
        res = np.sign(pick.ats) * side
        w, l = int((res > 0).sum()), int((res < 0).sum())
        pv = stats.binomtest(w, w + l, BREAKEVEN, alternative="greater").pvalue if w + l else 1
        out(f"  ATS picks |pred|>{thr}: {w}-{l} ({w / max(w + l, 1):.1%}), break-even 52.4%, p(beat)={pv:.3f}")

    # the scheduled-loss idea specifically: how do teams do ATS on a back-to-back vs a rested opponent?
    out("\n   schedule spots, ATS from the tired team's side (all seasons):")
    for name, mask, sign in (
        ("home on B2B, away rested", (d.h_b2b == 1) & (d.a_b2b == 0), 1),
        ("away on B2B, home rested", (d.a_b2b == 1) & (d.h_b2b == 0), -1),
        ("home 3-in-4, away not", (d.h_three_in_four == 1) & (d.a_three_in_four == 0), 1),
        ("away 3-in-4, home not", (d.a_three_in_four == 1) & (d.h_three_in_four == 0), -1),
    ):
        s = d[mask]
        cov = (np.sign(s.ats) * sign)
        w, l = int((cov > 0).sum()), int((cov < 0).sum())
        su = (np.sign(s.margin) * sign > 0).mean()
        out(f"     {name:<28} n={len(s):4d}  tired team straight-up win {su:.1%}  ATS {w}-{l} ({w / max(w + l, 1):.1%})"
            f"  avg ATS margin {(s.ats * sign).mean():+.2f}")


# ---------------------------------------------------------------- player

def player_table(data_dir, t):
    p = pd.read_csv(os.path.join(data_dir, "player_feats.csv"))
    p = p[(p.pgp >= 5) & (p.mpg >= 15) & p.base.notna()].copy()
    lg_drtg = t.groupby("season").drtg_r.mean()
    lg_pace = t.groupby("season").pace.mean()
    p["opp_def"] = p.opp_drtg - p.season.map(lg_drtg)        # + = opponent defends worse than average
    p["opp_pace"] = p.opp_pace - p.season.map(lg_pace)
    p["eff"] = (p.ppu - p.groupby("season").ppu.transform("mean")) / p.groupby("season").ppu.transform("std")
    p["usg_c"] = p.usg - p.groupby("season").usg.transform("mean")
    # "best scorer vs weakest positional defense": efficiency x positional weakness, and the team's
    # top-usage player at the position
    p["eff_x_dvp"] = p.eff * p.dvp
    top = p.groupby(["game_id", "team", "posb"]).usg.transform("max")
    p["top_at_pos"] = (p.usg == top).astype(int)
    p["top_x_dvp"] = p.top_at_pos * p.dvp
    # usage redistribution: missing teammates' usage, weighted by how much of the offense this player runs
    p["vac_x_usg"] = p.vac_usg * p.usg
    p["resid"] = p.pts - p.base
    return p.dropna(subset=["usg", "ppu", "opp_def", "dvp"])


PGROUPS = {
    "opponent team defense (current model's idea)": ["opp_def", "opp_pace"],
    "matchup: defense vs position": ["dvp"],
    "matchup: best scorer vs weakest positional D": ["dvp", "eff", "eff_x_dvp", "top_at_pos", "top_x_dvp"],
    "usage level": ["usg_c"],
    "usage: missing teammates' usage redistributed": ["vac_usg", "vac_x_usg", "vac_min"],
    "rest: back-to-back": ["b2b"],
}
PALL = sorted({c for v in PGROUPS.values() for c in v})


def player_tests(p, out):
    tr, te = p[p.season.isin(TRAIN)], p[p.season.isin(TEST)]
    out(f"\n\nPLAYER-GAMES (>= 5 games played, >= 15 mpg)  train {len(tr)}  test {len(te)}")
    out("Target: points minus the shrunk season average. Test-set RMSE of points; baseline = season average alone.")
    base_e = te.resid ** 2
    out(f"  {'baseline: shrunk season average':<52} RMSE {np.sqrt(base_e.mean()):.3f}  MAE {te.resid.abs().mean():.3f}")
    preds = {}
    for name, cols in list(PGROUPS.items()) + [("all linear", PALL)]:
        m, pr = fit_ols(tr, te, cols, "resid")
        preds[name] = pr
        e = (te.resid - pr) ** 2
        mean, lo, hi, pv = boot(base_e, e, te.date)
        out(f"  {name:<52} RMSE {np.sqrt(e.mean()):.3f}  dMSE {mean:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f}")
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                       l2_regularization=1.0, random_state=0)
    feats = PALL + ["base", "mpg", "ts"]
    gb.fit(tr[feats], tr.resid)
    pr = gb.predict(te[feats])
    e = (te.resid - pr) ** 2
    mean, lo, hi, pv = boot(base_e, e, te.date)
    out(f"  {'gradient boosting, all features (nonlinear check)':<52} RMSE {np.sqrt(e.mean()):.3f}  "
        f"dMSE {mean:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f}")

    out("\n   coefficients (all three seasons, HC3 robust), points above season average:")
    full = sm.OLS(p.resid, sm.add_constant(p[PALL])).fit(cov_type="HC3")
    for c in PALL:
        out(f"     {c:<12} {full.params[c]:+8.3f}  se {full.bse[c]:.3f}  p={full.pvalues[c]:.4f}")

    out("\n   does the signal pick overs? test set, over/under the player's season average (not a real prop line):")
    pr = preds["all linear"]
    for thr in (1.0, 2.0, 3.0):
        sel = np.abs(pr) > thr
        hit = (np.sign(te.resid[sel]) == np.sign(pr[sel]))[te.resid[sel] != 0]
        out(f"     |predicted edge| > {thr} pts: n={int(sel.sum()):5d}  correct side {hit.mean():.1%}")

    out("\n   matchup buckets (test set): actual points above season average")
    q = te.assign(dq=pd.qcut(te.dvp, 5, labels=["best D", "2", "3", "4", "worst D"]),
                  eq=pd.qcut(te.eff, 3, labels=["low eff", "mid eff", "high eff"]))
    tab = q.pivot_table(index="eq", columns="dq", values="resid", aggfunc="mean", observed=True).round(2)
    out("     " + tab.to_string().replace("\n", "\n     "))


def main(data_dir):
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    d, t = team_table(data_dir)
    team_tests(d, out)
    player_tests(player_table(data_dir, t), out)


if __name__ == "__main__":
    main(sys.argv[1])
