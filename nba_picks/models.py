from dataclasses import dataclass, field


@dataclass
class Split:
    """A per-team stat with home/away/overall splits (TeamRankings layout)."""
    overall: float = 0.0
    home: float = 0.0
    away: float = 0.0
    prev: float | None = None   # last season's overall value, the shrinkage prior

    def get(self, loc: str) -> float:
        return getattr(self, loc)


@dataclass
class Injury:
    name: str
    status: str


@dataclass
class Player:
    name: str
    team: str          # ESPN abbreviation, e.g. "BOS", "GS", "UTAH"
    ppg: float
    fga: float
    gp: int = 0        # games played this season


@dataclass
class Game:
    home: str          # full display name, e.g. "Boston Celtics"
    visitor: str
    home_abbr: str
    visitor_abbr: str
    spread: str = "N/A"

    @property
    def matchup(self) -> str:
        return f"{self.visitor} @ {self.home}"


@dataclass
class TeamContext:
    """Everything the model knows about one side of one game."""
    back_to_back: bool = False
    margin: float = 0.0            # location-specific scoring margin
    ats_cover_pct: float = 0.0     # location-specific cover %
    quarters: dict = field(default_factory=lambda: {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0})
    injuries: list = field(default_factory=list)   # list[Injury]
    opp_pts_allowed: float = 0.0   # this team's defense, overall
    # inputs to the fitted margin model (nba_picks/fitted.py)
    srs: float = 0.0               # schedule-adjusted net rating, points per 100 possessions
    missing_ppg: float = 0.0       # season points per game of regulars (20+ mpg) not playing
    usage_edge: float = 0.0        # active roster's projected points minus the team's season scoring


@dataclass
class TeamPick:
    game: Game
    score: float
    pick: str                      # team name or "Toss Up"
    confidence: int
    reasons: list

    @property
    def is_toss_up(self) -> bool:
        return self.pick == "Toss Up"


@dataclass
class MoneyLine:
    """American odds for one game, e.g. home=-150, away=+130."""
    home: str
    visitor: str
    home_odds: int
    visitor_odds: int
    book: str = ""


@dataclass
class PropLine:
    """A points over/under, e.g. line=23.5, over_odds=-115, under_odds=-105."""
    player: str
    line: float
    over_odds: int
    under_odds: int
    book: str = ""


@dataclass
class MarketEdge:
    """The model's view of one side of a market versus the book's price."""
    label: str           # "Boston Celtics ML" or "Jayson Tatum over 23.5"
    side: str            # team name, "over" or "under"
    odds: int
    model_prob: float    # model blended with the market
    market_prob: float   # no-vig implied probability
    edge: float          # model_prob - market_prob
    ev: float            # expected profit per 1 unit staked, at model_prob
    is_bet: bool
    detail: str = ""
    raw_model_prob: float = 0.0   # the model on its own, before blending


@dataclass
class PlayerProp:
    player: str
    team: str
    opponent: str
    base_ppg: float
    proj_pts: float
    confidence: int
    reasons: list
