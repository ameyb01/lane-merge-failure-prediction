#!/usr/bin/env python
"""
workzone/analysis/apply_fixes.py

Applies three fixes found by watching the recorded episodes on 2026-09-04.
Run once from the repo root:

    python workzone/analysis/apply_fixes.py

Each patch asserts its anchor text matches exactly once, so a partial or
repeated application fails loudly instead of corrupting a file.

  1. Per-vehicle collision sensors. The env had no collision detection at
     all, so two vehicles could rear-end each other and still be labelled
     'success'. CarlaBaseEnv.is_collision() is a single global flag from
     one actor's sensor and cannot attribute a hit, so every vehicle gets
     its own sensor with its own callback.
  2. Episodes could end while a vehicle was mid-merge, labelling it by
     whatever state it happened to be in.
  3. The closure camera only repositioned when nonego_location CHANGED.
     With a fixed work vehicle that is never, and the sensor respawns at
     the world origin between episodes.
"""

import re
import sys
from pathlib import Path

ENV = Path("car_dreamer/carla_workzone_env.py")
EVENTS = Path("workzone/taxonomy/events.py")
CAM = Path("car_dreamer/toolkit/observer/handlers/closure_camera_handler.py")


def patch(path: Path, old: str, new: str, label: str) -> None:
    src = path.read_text()
    n = src.count(old)
    if n != 1:
        sys.exit(f"FAILED [{label}]: anchor matched {n} times in {path}\n"
                 f"--- anchor ---\n{old}\n--------------")
    path.write_text(src.replace(old, new, 1))
    print(f"  ok  {label}")


print(f"patching {ENV}")

# ---- 1a. collision sensors: spawn one per vehicle -------------------
patch(
    ENV,
    """        # the observer needs an actor to attach to; this confers no
        # behavioural privilege -- it is a Driver like all the others""",
    """        self._spawn_collision_sensors()

        # the observer needs an actor to attach to; this confers no
        # behavioural privilege -- it is a Driver like all the others""",
    "1a call _spawn_collision_sensors",
)

# ---- 1b. the sensor helper itself -----------------------------------
patch(
    ENV,
    """    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego""",
    '''    def _spawn_collision_sensors(self) -> None:
        """
        One collision sensor per vehicle.

        CarlaBaseEnv.is_collision() reads a single global flag from the
        observer's own sensor, which cannot say WHICH vehicles collided.
        Attaching a sensor to each vehicle gives exact attribution and the
        collision partner, which matters for the rear-end cases: a merger
        who forces in and is struck is a different event from one that
        simply stops.
        """
        self._collisions = {}          # vehicle id -> list of event dicts
        self._collision_sensors = []

        bp = self._world.get_blueprint("sensor.other.collision")
        own_ids = {v.id for v in self.vehicles}

        for v in self.vehicles:
            sensor = self._world.spawn_unmanaged_actor(
                carla.Transform(), bp, attach_to=v)

            def on_hit(event, vid=v.id):
                other = event.other_actor
                other_id = other.id if other is not None else -1
                imp = event.normal_impulse
                self._collisions.setdefault(vid, []).append({
                    "tick": self._n_ticks,
                    "other_id": other_id,
                    "other_is_vehicle": other_id in own_ids,
                    "impulse": round(
                        (imp.x ** 2 + imp.y ** 2 + imp.z ** 2) ** 0.5, 2),
                })

            sensor.listen(on_hit)
            self._collision_sensors.append(sensor)

    def _destroy_collision_sensors(self) -> None:
        for s in getattr(self, "_collision_sensors", []):
            try:
                s.stop()
                s.destroy()
            except Exception:
                pass
        self._collision_sensors = []

    def get_ego_vehicle(self) -> carla.Actor:
        return self.ego''',
    "1b _spawn_collision_sensors helper",
)

# ---- 1c. tear the sensors down at reset -----------------------------
patch(
    ENV,
    """        self._load_geometry()

        self._episode_index = getattr(self, "_episode_index", -1) + 1""",
    """        # sensors from the previous episode are attached to actors the
        # WorldManager is about to destroy
        self._destroy_collision_sensors()

        self._load_geometry()

        self._episode_index = getattr(self, "_episode_index", -1) + 1""",
    "1c destroy sensors on reset",
)

# ---- 1d. init the collision dict with the rest of episode state -----
patch(
    ENV,
    """        self._resolved: Dict[int, str] = {}""",
    """        self._resolved: Dict[int, str] = {}
        self._collisions: Dict[int, list] = {}""",
    "1d init _collisions",
)

# ---- 1e. collision takes precedence in on_step ----------------------
patch(
    ENV,
    """        for d in self.drivers:
            update_stop_tracking(d, self.DT)
            if d.vehicle.id in self._resolved:
                continue
            outcome = classify_vehicle(d, self.TAPER_START_S, self.TAPER_END_S)""",
    """        for d in self.drivers:
            update_stop_tracking(d, self.DT)
            if d.vehicle.id in self._resolved:
                continue

            # A vehicle that crashed has not merged successfully, wherever
            # it ended up, so collision takes precedence. Only closed-lane
            # vehicles get an outcome; open-lane hits stay in the event log.
            if (self._collisions.get(d.vehicle.id)
                    and d.origin_lane_id == self.CLOSED_LANE):
                self._resolved[d.vehicle.id] = "collision"
                self._events.append((self._n_ticks, d.vehicle.id, "collision"))
                continue

            outcome = classify_vehicle(d, self.TAPER_START_S, self.TAPER_END_S)""",
    "1e collision precedence",
)

# ---- 1f. surface collisions in the logged info ----------------------
patch(
    ENV,
    """            "n_resolved": len(self._resolved),
            "outcomes": dict(self._resolved),""",
    """            "n_resolved": len(self._resolved),
            "outcomes": dict(self._resolved),
            "collisions": {k: list(v) for k, v in self._collisions.items()},
            "n_collisions": sum(len(v) for v in self._collisions.values()),""",
    "1f log collisions in info",
)

# ---- 2. do not end an episode mid-merge -----------------------------
patch(
    ENV,
    """        all_resolved = bool(closed) and all(
            d.vehicle.id in self._resolved for d in closed)

        cleared = bool(self.drivers) and all(
            d.progress() >= self.BUFFER_END_S for d in self.drivers)""",
    """        # A vehicle part-way across should finish. Without this an episode
        # can end while a merge is in progress, labelling that vehicle by
        # whatever state it happened to be in.
        mid_merge = any(d.state == MergeState.COMMITTED for d in self.drivers)

        all_resolved = (bool(closed)
                        and all(d.vehicle.id in self._resolved for d in closed)
                        and not mid_merge)

        cleared = (bool(self.drivers)
                   and all(d.progress() >= self.BUFFER_END_S for d in self.drivers)
                   and not mid_merge)""",
    "2 mid-merge terminal guard",
)

# ---- 3. taxonomy: add the COLLISION outcome -------------------------
print(f"patching {EVENTS}")
patch(
    EVENTS,
    '''class Outcome(Enum):
    """Mutually exclusive. Evaluated in order; first match wins."""''',
    '''class Outcome(Enum):
    """Mutually exclusive. Evaluated in order; first match wins."""
    COLLISION    = "collision"        # struck another vehicle; overrides all''',
    "3 COLLISION outcome",
)

# ---- 4. camera: position unconditionally ----------------------------
print(f"patching {CAM}")
cam_src = CAM.read_text()
old_block = re.search(
    r"        nonego_loc = env_state\.get\(\"nonego_location\", None\)\n"
    r"        if nonego_loc is not None and self\._camera is not None:\n"
    r"            if self\._nonego_location != nonego_loc:\n"
    r"                self\._nonego_location = nonego_loc\n"
    r"(                .*\n)+?"
    r"                self\._camera\.set_transform\(cam_transform\)\n",
    cam_src,
)
if old_block is None:
    sys.exit("FAILED [4]: could not locate the camera transform block")

new_block = '''        nonego_loc = env_state.get("nonego_location", None)
        if nonego_loc is not None and self._camera is not None:
            # Set unconditionally. Guarding on "has the location changed"
            # never fires when the work vehicle spawns in the same place
            # every episode, and the sensor is respawned at the world
            # origin between episodes, so the camera ends up looking at
            # nothing. Setting a transform is cheap.
            self._nonego_location = nonego_loc
            cam_transform = carla.Transform(
                carla.Location(
                    x=nonego_loc[0],
                    y=nonego_loc[1] + 3.0,   # just behind the truck
                    z=self._config.height,
                ),
                carla.Rotation(
                    pitch=self._config.pitch,
                    yaw=90,                  # facing oncoming traffic
                ),
            )
            self._camera.set_transform(cam_transform)
'''
CAM.write_text(cam_src.replace(old_block.group(0), new_block, 1))
print("  ok  4 camera positions unconditionally")

print("\nall patches applied")
