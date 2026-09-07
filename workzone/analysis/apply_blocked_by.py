#!/usr/bin/env python
"""
workzone/analysis/apply_blocked_by.py

Adds a `blocked_by` field to STOP_IN_LANE outcomes.

Roughly half of all stops are followers halted by the vehicle in front
rather than drivers who could not find a gap. Those are different
phenomena: predicting a blocked follower means predicting the vehicle
ahead of it, not this one. Recording which vehicle blocked it lets the two
be separated in analysis without changing any behaviour.

Run once from the repo root:

    python workzone/analysis/apply_blocked_by.py
"""

import sys
from pathlib import Path

ENV = Path("car_dreamer/carla_workzone_env.py")
COLLECT = Path("workzone/collect/collect_v2.py")


def patch(path: Path, old: str, new: str, label: str) -> None:
    src = path.read_text()
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED [{label}]: anchor matched {n} times in {path}\n"
                 f"--- anchor ---\n{old}\n--------------")
    path.write_text(src.replace(old, new, 1))
    print(f"  ok  {label}")


print(f"patching {ENV}")

# ---- 1. episode state -----------------------------------------------
patch(
    ENV,
    """        self._collisions: Dict[int, list] = {}""",
    """        self._collisions: Dict[int, list] = {}
        self._blocked: Dict[int, int] = {}""",
    "1 init _blocked",
)

# ---- 2. the helper ---------------------------------------------------
patch(
    ENV,
    """    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego""",
    '''    def _blocked_by(self, subject: Driver) -> int:
        """
        Id of a stopped vehicle within 12 m ahead in the same lane, or -1.

        A follower halted by the queue never made a merge decision, so its
        stop says nothing about gap acceptance. Recording the blocker keeps
        the two cases separable.
        """
        s_me = subject.progress()
        lane = subject.current_waypoint().lane_id
        best, best_d = -1, 1e9
        for other in self.drivers:
            if other is subject or other.speed() >= 0.5:
                continue
            if other.current_waypoint().lane_id != lane:
                continue
            d = other.progress() - s_me
            if 0 < d < 12.0 and d < best_d:
                best, best_d = other.vehicle.id, d
        return best

    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego''',
    "2 _blocked_by helper",
)

# ---- 3. record it when a stop is resolved ----------------------------
patch(
    ENV,
    """            if outcome is not None:
                self._resolved[d.vehicle.id] = outcome.value
                self._events.append((self._n_ticks, d.vehicle.id, outcome.value))""",
    """            if outcome is not None:
                self._resolved[d.vehicle.id] = outcome.value
                if outcome.value == "stop_in_lane":
                    self._blocked[d.vehicle.id] = self._blocked_by(d)
                self._events.append((self._n_ticks, d.vehicle.id, outcome.value))""",
    "3 record blocked_by on stop",
)

# ---- 4. surface it in info ------------------------------------------
patch(
    ENV,
    """            "collisions": {k: list(v) for k, v in self._collisions.items()},""",
    """            "collisions": {k: list(v) for k, v in self._collisions.items()},
            "blocked_by": dict(self._blocked),""",
    "4 log blocked_by in info",
)

# ---- 5. persist it in the manifest -----------------------------------
print(f"patching {COLLECT}")
patch(
    COLLECT,
    """        "collisions": {str(k): v for k, v in info.get("collisions", {}).items()},""",
    """        "collisions": {str(k): v for k, v in info.get("collisions", {}).items()},
        "blocked_by": {str(k): v for k, v in info.get("blocked_by", {}).items()},""",
    "5 save blocked_by to manifest",
)

print("\nall patches applied")
