"""Every weight and threshold the model uses, in one place.

Values match the original main.py so picks are directly comparable.
"""

# --- Team pick weights (positive score favors home, negative favors visitor) ---
REST_EDGE = 3.0              # one team on a back-to-back, the other rested

STAR_PPG = 25.0              # an Out/Doubtful player above this is a "star"
STAR_PENALTY = 5.0
KEY_PPG = 15.0               # above this (and not a star) is a "key" player
KEY_PENALTY = 3.0
OUT_STATUSES = ("Out", "Doubtful")

MARGIN_THRESHOLD = 3.0       # home-split margin minus away-split margin
MARGIN_EDGE = 1.5

ATS_THRESHOLD = 5.0          # cover % difference
ATS_EDGE = 1.0

QUARTER_WEIGHTS = {"q1": 0.3, "q2": 0.1, "q3": 0.4, "q4": 0.2}
QUARTER_THRESHOLD = 1.5
QUARTER_EDGE = 1.0

# --- Pick and confidence ---
TOSS_UP_BAND = 0.5           # |score| at or below this is a toss up
CONFIDENCE_SCALE = 8.0       # |score| of 8 maps to 99%
CONFIDENCE_CAP = 99

# Confidence tiers for the track record, as (label, lowest confidence in tier)
TIERS = (("High (76-100%)", 76), ("Med (50-75%)", 50), ("Low (0-49%)", 0))

# --- Player props ---
LEAGUE_AVG_PTS_ALLOWED = 115.0
BAD_DEFENSE = 118.0          # opponent allows more than this: +conf
GOOD_DEFENSE = 110.0         # opponent allows less than this: -conf
BAD_DEFENSE_CONF = 20
GOOD_DEFENSE_CONF = -10

PROJ_LOSS_FACTOR = 0.95
PROJ_WIN_FACTOR = 1.02
PROJ_WIN_CONF = 10

USAGE_BUMP_FACTOR = 1.15     # a scoring teammate is Out/Doubtful
USAGE_BUMP_CONF = 15

HIGH_VOLUME_FGA = 18.0
HIGH_VOLUME_CONF = 10

PROP_BASE_CONF = 50
PROP_MIN_CONF = 70           # only report props at or above this
PROP_CONF_CAP = 99           # the factors can add up past 100

# --- Market comparison ---
MIN_EDGE = 0.03              # model prob minus no-vig market prob needed to flag a bet
# The model gives a point projection, not a distribution. To price over/under we
# assume actual points ~ Normal(projection, sd), sd = A + B * sqrt(projection).
# Fitted on 55k player-games (research/fit_models.py): sd is ~6.7 at 14 points, ~8.6 at 27.
PROP_SD_A = 1.373
PROP_SD_B = 1.408

# --- Calibration ---
# Early-season shrinkage: blended = (n * current + k * prior) / (n + k), n = games played.
# k is "how many games the prior is worth". At n == k the two count equally.
# Priors are last season's value where we have it, else the league default.
SHRINK_K_TEAM = 10           # margins, quarter margins, points allowed
SHRINK_K_ATS = 25            # cover % is mostly noise, so it regresses hard toward 50
SHRINK_K_PLAYER = 3          # player points per game; k = 2-4 was best out of sample (8 was a guess)
LEAGUE_ATS_PCT = 50.0
LEAGUE_MARGIN = 0.0

# Market blending, done in log-odds: blended = w * model + (1 - w) * market.
# The closing line is the best public predictor, so the model only gets a minority say.
# Team: on 2025-26 out of sample, log loss was best at w = 0-0.1 (the fitted margin model added
# nothing significant to the moneyline, p = 0.60), so 0.10 is an upper bound, not an edge.
# Props: no usable history of prop lines to fit against, so 0.30 is still a guess.
TEAM_MODEL_WEIGHT = 0.10
PROP_MODEL_WEIGHT = 0.30
PROB_CLIP = 0.01             # keep probabilities inside [0.01, 0.99] before taking log-odds
# If the raw model and the market differ by more than this, assume bad inputs, not an edge.
MAX_DISAGREEMENT = 0.25
