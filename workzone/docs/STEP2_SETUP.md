# Step 2 setup

Three files, three registration edits, then a smoke test.

## 1. File placement

    car_dreamer/carla_workzone_env.py        <- new env (from carla_workzone_env.py)
    workzone/taxonomy/__init__.py            <- empty
    workzone/taxonomy/events.py              <- taxonomy (from events.py)
    workzone/__init__.py                     <- empty
    workzone/driver/__init__.py              <- empty

The `__init__.py` files matter: the env imports
`from workzone.driver.driver import Driver`, which needs `workzone` and
`workzone.driver` to be importable packages.

    cd ~/research2/CarDreamer
    mkdir -p workzone/taxonomy
    touch workzone/__init__.py \
          workzone/driver/__init__.py \
          workzone/taxonomy/__init__.py

## 2. Register the env class

In `car_dreamer/__init__.py`, alongside the existing workzone imports:

```python
from .carla_workzone_env import CarlaWorkzoneEnv
```

## 3. Register the task

Append to `car_dreamer/configs/tasks.yaml`. Note it does NOT use the
`<<: *carla_wpt` anchor -- that anchor carries waypoint-following
assumptions this task must not inherit.

```yaml
carla_workzone:
  env:
    world:
      town: Town04
      fixed_delta_seconds: 0.1
      actor_active_distance: 500      # corridor is ~445 m; 300 is too short
    name: CarlaWorkzoneEnv-v0
    observation.enabled: [collision, closure_camera]
    num_vehicles: 0

    # geometry -- see workzone/configs/geometry.yaml
    closed_lane_id:   -1
    open_lane_id:     -2
    lane_ref_x:       5.24
    approach_start_y: 230.0
    taper_start_y:      0.0
    taper_end_y:     -165.0
    buffer_end_y:    -215.0

    # traffic
    posted_speed:          20.12      # m/s, 45 mph
    n_vehicles:            12
    closed_lane_fraction:  0.5
    min_spawn_headway:     18.0       # m between spawn slots
    settle_ticks:          10
    step_cap:              800
    episode_seed:          0

    action:
      discrete: False
      continuous_acc:   [-3.0, 3.0]
      continuous_steer: [-1.0, 1.0]

    terminal:
      time_limit: 800

  dreamerv3:
    run.log_keys_video: [closure_camera]
```

The `action` block is required because `CarlaBaseEnv._get_action_space`
reads it, and `step(action)` expects an argument. The value is ignored --
every vehicle generates its own control.

## 4. Raise actor_active_distance in common.yaml

```yaml
    actor_active_distance: 500
```

Currently 300, which is shorter than the 445 m corridor.

## 5. Smoke test

```bash
cd ~/research2/CarDreamer
python -c "
import car_dreamer
env, cfg = car_dreamer.create_task('carla_workzone', [])
obs = env.reset()
print('reset ok, obs keys:', list(obs.keys()))
for i in range(300):
    obs, reward, done, info = env.step(env.action_space.sample())
    if i % 50 == 0:
        print(i, 'resolved', info.get('n_resolved'), '/', info.get('n_vehicles'))
    if done:
        print('terminated at', i, info.get('outcomes'))
        break
"
```

## Known unknowns

- **`WorldManager.reset()` actor cleanup.** Not yet read. If it does not
  destroy actors between episodes, the second episode will spawn on top
  of the first. Check with `grep -n "def reset" -A 40
  car_dreamer/toolkit/carla_manager/world_manager.py` before running
  multiple episodes.
- **`self._world.step()` inside `on_reset`.** Used for the settle ticks.
  If `WorldManager.step()` re-enters `on_step`, this will recurse. If the
  smoke test hangs or errors on reset, that is the cause -- swap it for
  `self._world.carla_world.tick()`.
- **`stop_ticks` is not yet wired.** `update_stop_tracking` exists in the
  taxonomy but nothing calls it, so `STOP_IN_LANE` cannot fire yet. Add
  the call in `on_step` once the smoke test passes.
