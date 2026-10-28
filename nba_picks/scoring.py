"""The analysis itself: pure functions, no network, no files.

Team score convention: positive favors the home team, negative the visitor.
"""
from . import config as C
from .models import Game, Player, PlayerProp, TeamContext, TeamPick


def short_name(full: str) -> str:
    if "Portland" in full:
        return "Portland"
    if "New York" in full:
        return "New York"
    return full.split(" ")[-1]


def quarter_score(quarters: dict) -> float:
    return sum(quarters.get(q, 0.0) * w for q, w in C.QUARTER_WEIGHTS.items())


def injury_penalty(injuries, players_by_name: dict) -> tuple[float, list]:
    """Penalty for a team's Out/Doubtful scorers, plus a note for each."""
    penalty, notes = 0.0, []
    for inj in injuries:
        if inj.status not in C.OUT_STATUSES:
            continue
        player = lookup_player(inj.name, players_by_name)
        ppg = player.ppg if player else 0.0
        if ppg > C.STAR_PPG:
            penalty += C.STAR_PENALTY
            notes.append(f"STAR OUT: {inj.name}")
        elif ppg > C.KEY_PPG:
            penalty += C.KEY_PENALTY
            notes.append(f"KEY OUT: {inj.name}")
    return penalty, notes


def lookup_player(name: str, players_by_name: dict):
    if name in players_by_name:
        return players_by_name[name]
    for key, p in players_by_name.items():   # injury report vs stats page spelling
        if name in key:
            return p
    return None


def confidence_from_score(score: float) -> int:
    mag = abs(score)
    if mag <= C.TOSS_UP_BAND:
        return 50
    return int(min(C.CONFIDENCE_CAP, 50 + (mag / C.CONFIDENCE_SCALE) * 49))


def _edge(diff, threshold, weight, home, vis, strong_msg, slight_msg=None):
    """Shared shape of the margin / ATS / quarter factors."""
    if diff > threshold:
        return weight, strong_msg.format(team=home, diff=round(diff, 1))
    if diff < -threshold:
        return -weight, strong_msg.format(team=vis, diff=round(-diff, 1))
    if slight_msg is None:
        return 0.0, None
    return 0.0, slight_msg.format(team=home if diff > 0 else vis, diff=round(abs(diff), 1))


def score_game(game: Game, home: TeamContext, vis: TeamContext, players_by_name: dict) -> TeamPick:
    h, v = short_name(game.home), short_name(game.visitor)
    score, reasons = 0.0, []

    # 1. Rest
    if vis.back_to_back and not home.back_to_back:
        score += C.REST_EDGE
        reasons.append(f"Rest favors {h} ({v} on B2B).")
    elif home.back_to_back and not vis.back_to_back:
        score -= C.REST_EDGE
        reasons.append(f"Rest favors {v} ({h} on B2B).")
    elif home.back_to_back:
        reasons.append("Both teams on a back-to-back.")
    else:
        reasons.append("Both teams are rested.")

    # 2. Injuries: a hurt visitor helps home and vice versa
    vis_pen, vis_notes = injury_penalty(vis.injuries, players_by_name)
    home_pen, home_notes = injury_penalty(home.injuries, players_by_name)
    score += vis_pen - home_pen
    if vis_notes:
        reasons.append(f"Vis Injury: {', '.join(vis_notes)}.")
    if home_notes:
        reasons.append(f"Home Injury: {', '.join(home_notes)}.")

    # 3. Scoring margin (home team's home split vs visitor's road split)
    pts, msg = _edge(home.margin - vis.margin, C.MARGIN_THRESHOLD, C.MARGIN_EDGE, h, v,
                     "{team} dominates margin (+{diff}).", "{team} has slight margin edge (+{diff}).")
    score += pts
    reasons.append(msg)

    # 4. Against the spread
    ats_diff = home.ats_cover_pct - vis.ats_cover_pct
    pts, msg = _edge(ats_diff, C.ATS_THRESHOLD, C.ATS_EDGE, h, v, "{team} covers better (+{diff}%).")
    score += pts
    reasons.append(msg or f"ATS trends are comparable ({home.ats_cover_pct:.1f}% vs {vis.ats_cover_pct:.1f}%).")

    # 5. Quarter-by-quarter margin, weighted toward Q3 and Q1
    pts, msg = _edge(quarter_score(home.quarters) - quarter_score(vis.quarters), C.QUARTER_THRESHOLD,
                     C.QUARTER_EDGE, h, v, "{team} has stronger quarter trends (+{diff}).")
    score += pts
    if msg:
        reasons.append(msg)

    if score > C.TOSS_UP_BAND:
        pick = game.home
    elif score < -C.TOSS_UP_BAND:
        pick = game.visitor
    else:
        pick = "Toss Up"
    return TeamPick(game, score, pick, confidence_from_score(score), reasons)


def project_player(player: Player, my_team: str, opponent: str, team_pick: TeamPick,
                   opp_pts_allowed: float, teammate_injuries, players_by_name: dict) -> PlayerProp:
    opp_def = opp_pts_allowed or C.LEAGUE_AVG_PTS_ALLOWED
    proj = player.ppg * (opp_def / C.LEAGUE_AVG_PTS_ALLOWED)
    conf, reasons = C.PROP_BASE_CONF, []

    if opp_def > C.BAD_DEFENSE:
        conf += C.BAD_DEFENSE_CONF
        reasons.append("Bad Def")
    elif opp_def < C.GOOD_DEFENSE:
        conf += C.GOOD_DEFENSE_CONF

    if team_pick.pick == my_team:
        proj *= C.PROJ_WIN_FACTOR
        conf += C.PROJ_WIN_CONF
    elif not team_pick.is_toss_up:
        proj *= C.PROJ_LOSS_FACTOR
        reasons.append("Proj Loss")

    out = [i.name for i in teammate_injuries
           if i.status in C.OUT_STATUSES and i.name != player.name and lookup_player(i.name, players_by_name)]
    if out:
        proj *= C.USAGE_BUMP_FACTOR
        conf += C.USAGE_BUMP_CONF
        reasons.append(f"Usage Bump ({len(out)} Out)")

    if player.fga > C.HIGH_VOLUME_FGA:
        conf += C.HIGH_VOLUME_CONF

    return PlayerProp(player.name, my_team, opponent, player.ppg, round(proj, 1),
                      min(conf, C.PROP_CONF_CAP), reasons)


def player_props(picks, contexts: dict, players: list, injuries_by_team: dict,
                 min_conf: int = C.PROP_MIN_CONF) -> list:
    """Props at or above min_conf for every scored player in today's games.

    contexts maps full team name -> TeamContext; injuries_by_team maps full team name -> [Injury].
    """
    players_by_name = {p.name: p for p in players}
    props = []
    for tp in picks:
        g = tp.game
        sides = {g.home_abbr: (g.home, g.visitor), g.visitor_abbr: (g.visitor, g.home)}
        for p in players:
            if p.team not in sides:
                continue
            mine, opp = sides[p.team]
            if any(i.name == p.name and i.status in C.OUT_STATUSES for i in injuries_by_team.get(mine, [])):
                continue   # no prop for a player who isn't playing
            prop = project_player(p, mine, opp, tp, contexts[opp].opp_pts_allowed,
                                  injuries_by_team.get(mine, []), players_by_name)
            if prop.confidence >= min_conf:
                props.append(prop)
    return sorted(props, key=lambda x: (-x.confidence, -x.proj_pts))


# --- Grading a pick log ---

def grade_pick(pick: str, winner: str) -> str:
    if pick == "Toss Up":
        return "Skipped"
    return "Win" if pick == winner else "Loss"


def tier_of(confidence: int) -> str:
    return next(label for label, floor in C.TIERS if confidence >= floor)


def tier_stats(rows: list[dict]) -> dict[str, dict]:
    """Hit rate by confidence tier. rows: dicts with "Confidence" and "Result" (Win/Loss/...)."""
    stats = {label: {"wins": 0, "total": 0} for label, _ in C.TIERS}
    for r in rows:
        if r["Result"] not in ("Win", "Loss"):
            continue
        try:
            t = tier_of(int(float(r["Confidence"])))
        except ValueError:
            continue
        stats[t]["total"] += 1
        stats[t]["wins"] += r["Result"] == "Win"
    return stats
