"""Versioned season-end Elo state and offseason regression."""

import math

ELO_VERSION = "d1-k20-home65-v1"


def season_initial_ratings(eligible, season, prior_state=None, carryover=0.75, bootstrap=False):
    if not math.isfinite(carryover) or not 0 <= carryover <= 1:
        raise ValueError("Elo carryover must be between 0 and 1")
    if prior_state is None:
        if not bootstrap:
            raise ValueError("Provide prior-season Elo state or explicitly bootstrap Elo")
        return {t: 1500.0 for t in eligible}
    if bootstrap:
        raise ValueError("Choose prior-season Elo or bootstrap, not both")
    if prior_state["season"] != season - 1:
        raise ValueError("Elo state must come from the immediately previous season")
    if prior_state["elo_version"] != ELO_VERSION:
        raise ValueError("Incompatible Elo state version")
    if prior_state["through_date_exclusive"] != f"{season - 1}-07-01":
        raise ValueError("Elo state must represent a completed prior season")
    prior = {int(t): float(v) for t, v in prior_state["ratings"].items()}
    if not prior or any(not math.isfinite(v) for v in prior.values()):
        raise ValueError("Invalid prior Elo ratings")
    return {t: 1500.0 + carryover * (prior.get(t, 1500.0) - 1500.0) for t in eligible}
