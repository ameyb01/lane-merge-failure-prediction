"""
workzone/taxonomy/events.py

The failure taxonomy. Four mutually exclusive primary outcomes plus
secondary flags that may co-occur with any of them.

Positional thresholds are fractions of taper length L, not metres, so the
taxonomy survives a change of posted speed or lane width.

Every constant marked [ASSUMED] needs a citation or a sensitivity sweep.
See PROJECT_MEMO.md sections 4.5 and 9.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


# ---- thresholds -------------------------------------------------

STOP_SPEED = 0.5          # m/s   [ASSUMED]
STOP_DURATION = 2.0       # s     [ASSUMED]
FORCED_FRACTION = 0.25    # of L  [ASSUMED]

TTC_CONFLICT = 1.5        # s     [ASSUMED] verify vs surrogate-safety work
HARD_BRAKE = 3.4          # m/s^2 [ASSUMED] ~0.35 g


class Outcome(Enum):
    """Mutually exclusive. Evaluated in order; first match wins."""
    COLLISION    = "collision"        # struck another vehicle; overrides all
    STOP_IN_LANE = "stop_in_lane"     # halted in the closed lane
    NO_MERGE     = "no_merge"         # reached taper_end still in the closed lane
    FORCED_MERGE = "forced_merge"     # completed inside the final 0.25 L
    SUCCESS      = "success"          # completed before that


class Flag(Enum):
    """May co-occur with any outcome. NOT terminal conditions."""
    FOLLOWER_CONFLICT   = "follower_conflict"
    FOLLOWER_HARD_BRAKE = "follower_hard_brake"


def forced_merge_boundary(taper_start_s: float, taper_end_s: float) -> float:
    """s beyond which a completed merge counts as forced."""
    L = taper_end_s - taper_start_s
    return taper_end_s - FORCED_FRACTION * L


def classify_vehicle(driver, taper_start_s: float,
                     taper_end_s: float) -> Optional[Outcome]:
    """
    Return this vehicle's outcome, or None if it is not yet resolved.

    Vehicles that started in the open lane never resolve -- they have no
    merge to succeed or fail at.
    """
    from workzone.driver.driver import MergeState

    if driver.state == MergeState.NOT_MERGING:
        return None

    if driver.state == MergeState.MERGED:
        s = driver.merge_complete_progress
        if s is None:
            return None
        boundary = forced_merge_boundary(taper_start_s, taper_end_s)
        return Outcome.FORCED_MERGE if s >= boundary else Outcome.SUCCESS

    # A COMMITTED vehicle is mid-manoeuvre, not queued. If it is also
    # stationary, that is a controller deadlock rather than a driver
    # choosing to stop, and labelling it STOP_IN_LANE hides the bug.
    in_closed = driver.current_waypoint().lane_id == driver.origin_lane_id
    if (in_closed and driver.state != MergeState.COMMITTED
            and getattr(driver, "stop_ticks", 0) * 0.1 >= STOP_DURATION):
        return Outcome.STOP_IN_LANE
    if driver.progress() >= taper_end_s:
        return Outcome.NO_MERGE

    return None


def update_stop_tracking(driver, dt: float) -> None:
    """
    Accumulate stationary time. Called once per tick per driver.

    Kept out of Driver itself so the driver model stays a behaviour model
    and the taxonomy stays a labelling concern.
    """
    if driver.speed() < STOP_SPEED:
        driver.stop_ticks = getattr(driver, "stop_ticks", 0) + 1
    else:
        driver.stop_ticks = 0
