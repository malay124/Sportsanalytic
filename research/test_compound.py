"""Do the small situational edges add up to a big swing when they stack?

    python research/test_compound.py DATA_DIR > research/results_compound.txt

Small effects that are each insignificant can still matter together, so they are combined three ways:
  1. A-priori score: every factor z-scored and summed with the sign the theory predicts. Nothing is fit to
     outcomes, so all three seasons are fair test data and there is no overfitting.
  2. Fitted score: OLS weights learned on 2023-24 + 2024-25 only, judged on 2025-26.
  3. Stack count: how many factors favor the same side by at least half a standard deviation.
Each is judged against the margin (does the swing exist?) and against the closing spread (is it priced?).
Super-additivity is tested with a squared term: if stacking compounds, extreme scores should beat a line.
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from test_hypotheses import BREAKEVEN, TEST, TRAIN, boot, player_table, team_table

# factor -> sign the theory predicts for the home team (home minus away difference)
SITUATIONAL = {
    "d_b2b": -1, "d_three_in_four": -1, "d_five_in_seven": -1, "d_rest_days": +1, "d_road_b2b": -1,
    "d_sos": +1,                              # tougher schedule so far: better than the record says
    "d_bench_net": +1, "d_bench_ts": +1, "d_bench_share": +1, "d_depth": +1,
    "d_vac_ppg": -1, "d_usage_edge": +1,
}
NO_AVAIL = [k for k in SITUATIONAL if k not in ("d_vac_ppg", "d_usage_edge")]


def apriori(d, cols, ref):
    z = (d[cols] - ref[cols].mean()) / ref[cols].std()
    return (z * pd.Series({c: SITUATIONAL[c] for c in cols})).sum(axis=1)


def stack_count(d, cols, ref):
    z = (d[cols] - ref[cols].mean()) / ref[cols].std() * pd.Series({c: SITUATIONAL[c] for c in cols})
    return (z > 0.5).sum(axis=1) - (z < -0.5).sum(axis=1)   # net number of factors favoring home


def ats_record(sub, side):
    res = np.sign(sub.ats) * side
    w, l = int((res > 0).sum()), int((res < 0).sum())
    p = stats.binomtest(w, w + l, BREAKEVEN, alternative="greater").pvalue if w + l else 1.0
    return w, l, p


def judge(d, score, name, out, train_mask=None):
    d = d.assign(s=score)
    out(f"\n== {name}")
    m = sm.OLS(d.margin, sm.add_constant(d[["d_srs", "s"]])).fit(cov_type="HC3")
    a = sm.OLS(d.ats, sm.add_constant(d[["s"]])).fit(cov_type="HC3")
    q = sm.OLS(d.ats, sm.add_constant(d[["s"]].assign(s2=d.s ** 2 * np.sign(d.s)))).fit(cov_type="HC3")
    sd = d.s.std()
    out(f"  margin, holding team rating fixed: {m.params['s'] * sd:+.2f} pts per 1 sd of score (p={m.pvalues['s']:.4f})")
    out(f"  vs closing spread:                 {a.params['s'] * sd:+.2f} pts per 1 sd of score (p={a.pvalues['s']:.4f})")
    out(f"  compounding (signed square term) vs spread: p={q.pvalues['s2']:.3f}")
    out("  by decile of the score (home perspective):")
    d["dec"] = pd.qcut(d.s.rank(method="first"), 10, labels=False)
    for k, g in d.groupby("dec"):
        side = 1 if g.s.mean() > 0 else -1
        w, l, p = ats_record(g, side)
        out(f"    {k + 1:>2}: score {g.s.mean():+6.2f}  n={len(g):4d}  margin {g.margin.mean():+6.2f}  "
            f"spread said {-g.spread.mean():+6.2f}  beat spread by {g.ats.mean():+5.2f}  "
            f"ATS backing the favored side {w}-{l} ({w / max(w + l, 1):.1%}) p={p:.3f}")
    for tail in (0.10, 0.05, 0.02):
        lo, hi = d.s.quantile(tail), d.s.quantile(1 - tail)
        g = d[(d.s <= lo) | (d.s >= hi)]
        side = np.where(g.s > 0, 1, -1)
        res = np.sign(g.ats) * side
        w, l = int((res > 0).sum()), int((res < 0).sum())
        p = stats.binomtest(w, w + l, BREAKEVEN, alternative="greater").pvalue
        swing = (g.margin * side).mean() - (-g.spread * side).mean()
        out(f"  most extreme {tail:.0%} both tails: n={len(g)}  ATS {w}-{l} ({w / (w + l):.1%}) p={p:.3f}  "
            f"avg result beyond the spread {swing:+.2f} pts")


def team(data_dir, out):
    d, _ = team_table(data_dir)
    d = d.copy()
    ref = d[d.season.isin(TRAIN)]
    out(f"TEAM GAMES: {len(d)} (all three seasons; a-priori scores fit nothing to outcomes)")
    judge(d, apriori(d, list(SITUATIONAL), ref), "1a. a-priori score, all 12 situational factors", out)
    judge(d, apriori(d, NO_AVAIL, ref), "1b. a-priori score, schedule + SOS + bench only (no injuries)", out)

    out("\n== 3. stack count: net number of factors favoring home by > 0.5 sd (all 12 factors)")
    d["n"] = stack_count(d, list(SITUATIONAL), ref).clip(-5, 5)
    for k, g in d.groupby("n"):
        side = 1 if k > 0 else -1
        w, l, p = ats_record(g, side) if k else (0, 0, 1.0)
        rec = f"ATS backing the stacked side {w}-{l} ({w / max(w + l, 1):.1%}) p={p:.3f}" if k else ""
        out(f"    {k:+d}: n={len(g):4d}  margin {g.margin.mean():+6.2f}  spread said {-g.spread.mean():+6.2f}  {rec}")

    tr, te = d[d.season.isin(TRAIN)], d[d.season.isin(TEST)]
    cols = list(SITUATIONAL)
    out("\n== 2. fitted score: weights learned on 2023-25, judged on 2025-26 only")
    w = sm.OLS(tr.ats, sm.add_constant(tr[cols])).fit()
    out("  learned weights (pts per unit): " + ", ".join(f"{c[2:]} {w.params[c]:+.2f}" for c in cols))
    te = te.assign(s=w.predict(sm.add_constant(te[cols])))
    e0, e1 = te.ats ** 2, (te.ats - te.s) ** 2
    mean, lo, hi, p = boot(e0, e1, te.date)
    out(f"  vs 'spread is right': MSE {mean:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}  (negative would mean it helps)")
    for tail in (0.2, 0.1, 0.05):
        cut = te.s.abs().quantile(1 - tail)
        g = te[te.s.abs() >= cut]
        res = np.sign(g.ats) * np.sign(g.s)
        wn, ln = int((res > 0).sum()), int((res < 0).sum())
        pv = stats.binomtest(wn, wn + ln, BREAKEVEN, alternative="greater").pvalue
        out(f"  strongest {tail:.0%} of fitted signals: {wn}-{ln} ({wn / (wn + ln):.1%}) p={pv:.3f}")
    # the same, with ridge-like shrinkage via fewer degrees of freedom: equal weights fit by one scalar
    s_tr = apriori(tr, cols, tr)
    k = sm.OLS(tr.ats, sm.add_constant(s_tr)).fit().params.iloc[1]
    te_s = apriori(te, cols, tr) * k
    e2 = (te.ats - te_s) ** 2
    mean, lo, hi, p = boot(e0, e2, te.date)
    out(f"  equal-weight score scaled on train ({k:+.3f} pts/unit): MSE {mean:+.2f} [{lo:+.2f},{hi:+.2f}] p={p:.3f}")


def players(data_dir, out):
    _, t = team_table(data_dir)
    p = player_table(data_dir, t)
    out("\n\nPLAYERS: do matchup + usage + pace + defense edges compound?")
    signs = {"opp_def": 1, "opp_pace": 1, "dvp": 1, "eff_x_dvp": 1, "top_x_dvp": 1, "vac_min": 1, "vac_x_usg": 1}
    ref = p[p.season.isin(TRAIN)]
    z = (p[list(signs)] - ref[list(signs)].mean()) / ref[list(signs)].std()
    p = p.assign(s=z.sum(axis=1), n=(z > 0.5).sum(axis=1) - (z < -0.5).sum(axis=1))
    m = sm.OLS(p.resid, sm.add_constant(p[["s"]].assign(s2=p.s ** 2 * np.sign(p.s)))).fit(cov_type="HC3")
    out(f"  {m.params['s'] * p.s.std():+.2f} pts per 1 sd of the a-priori score (p={m.pvalues['s']:.4f}); "
        f"compounding term p={m.pvalues['s2']:.3f} ({m.params['s2']:+.3f})")
    out("  net count of favorable factors -> actual points vs season average:")
    for k, g in p.groupby(p.n.clip(-4, 5)):
        out(f"    {k:+d}: n={len(g):6d}  points vs average {g.resid.mean():+.2f}  over the average {(g.resid > 0).mean():.1%}")


def main(data_dir):
    out = print
    team(data_dir, out)
    players(data_dir, out)


if __name__ == "__main__":
    main(sys.argv[1])
