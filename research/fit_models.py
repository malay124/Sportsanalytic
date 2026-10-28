"""Fit the coefficients that nba_picks/fitted.py hard-codes, using only the features that survived
test_hypotheses.py. Refit on all three seasons once the out-of-sample check has passed.

    python research/fit_models.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, __file__.rsplit("research", 1)[0])
from test_hypotheses import player_table, team_table  # noqa: E402

TEAM_FEATURES = ["d_srs", "d_b2b", "d_vac_ppg", "d_usage_edge"]
PLAYER_FEATURES = ["opp_def", "opp_pace", "dvp", "eff", "eff_x_dvp", "top_at_pos", "top_x_dvp", "usg_c",
                   "vac_min", "vac_x_usg"]


def main(data_dir):
    d, t = team_table(data_dir)
    d = d[d.neutral == 0]
    m = sm.OLS(d.margin, sm.add_constant(d[TEAM_FEATURES])).fit(cov_type="HC3")
    print("TEAM margin model (intercept = home court, points):")
    for k, v in m.params.items():
        print(f"  {k:<14} {v:+.4f}  (se {m.bse[k]:.4f})")
    print(f"  residual sd {np.sqrt(m.mse_resid):.3f}")

    p = player_table(data_dir, t)
    m = sm.OLS(p.resid, sm.add_constant(p[PLAYER_FEATURES])).fit(cov_type="HC3")
    print("\nPLAYER points-above-baseline model:")
    for k, v in m.params.items():
        print(f"  {k:<12} {v:+.4f}  (se {m.bse[k]:.4f})")
    print("  league means used for centering: usg", round(p.usg.mean(), 4), " ppu", round(p.ppu.mean(), 4),
          " ppu sd", round(p.ppu.std(), 4))

    # spread of actual points around the projection, by projection size (for prop over/under pricing)
    p = p.assign(fit=m.fittedvalues, err=p.resid - m.fittedvalues, proj=p.base + m.fittedvalues)
    bins = pd.cut(p.proj, [0, 8, 12, 16, 20, 25, 30, 45])
    sd = p.groupby(bins, observed=True).err.std()
    mid = p.groupby(bins, observed=True).proj.mean()
    print("\n  points sd by projection:", {round(a, 1): round(b, 2) for a, b in zip(mid, sd)})
    fit = np.polyfit(np.sqrt(mid), sd, 1)
    print(f"  sd ~= {fit[1]:.3f} + {fit[0]:.3f} * sqrt(projection)")

    # shrinkage k for player ppg: which k predicts next-game points best (out of sample season)?
    from nba_picks.calibrate import shrink
    q = pd.read_csv(f"{data_dir}/player_feats.csv")
    q = q[(q.pgp >= 1) & (q.mpg >= 15) & q.prior_ppg.notna() & (q.season == "2025-26")]
    print("\n  player shrinkage k (2025-26, players with a prior season):")
    for k in (0, 2, 4, 8, 12, 20, 30):
        b = [shrink(v, n, pr, k) for v, n, pr in zip(q.ppg_raw, q.pgp, q.prior_ppg)]
        print(f"    k={k:<3} RMSE {np.sqrt(np.mean((q.pts - b) ** 2)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
