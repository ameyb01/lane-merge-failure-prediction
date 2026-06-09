from typing import Dict, Tuple

import carla
import numpy as np
from gym import spaces

from ...carla_manager import WorldManager
from .base_handler import BaseHandler


class ClosureCameraHandler(BaseHandler):
    """
    Fixed camera at the lane closure (Cybertruck) position,
    looking back toward approaching traffic.

    Simulates a roadside sensor fixed at the point of closure
    observing approaching vehicles. Position is updated each
    episode reset based on the Cybertruck spawn location passed
    through env_state["nonego_location"].

    Config parameters:
        key: observation key (default: closure_camera)
        shape: image shape [H, W, 3] (default: [128, 128, 3])
        height: camera height above road (default: 3.0)
        pitch: camera pitch angle (default: -10)
        fov: field of view (default: 90)
        sensor_tick: sensor update rate (default: 0.0)
    """

    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._camera = None
        self._data = None
        self._nonego_location = None

    def get_observation_space(self) -> Dict:
        return {
            self._config.key: spaces.Box(
                low=0, high=255,
                shape=self._config.shape,
                dtype=np.uint8
            )
        }

    def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
        # Update camera position if nonego location changed
        nonego_loc = env_state.get("nonego_location", None)
        if nonego_loc is not None and self._camera is not None:
            if self._nonego_location != nonego_loc:
                self._nonego_location = nonego_loc
                cam_transform = carla.Transform(
                    carla.Location(
                        x=nonego_loc[0],
                        y=nonego_loc[1] + 3.0,  # slightly behind truck toward ego
                        z=self._config.height,
                    ),
                    carla.Rotation(
                        pitch=self._config.pitch,
                        yaw=90,  # looking toward y=130 where ego spawns
                    ),
                )
                self._camera.set_transform(cam_transform)

        obs_data = self._data if self._data is not None else np.zeros(
            self._config.shape, dtype=np.uint8
        )
        return {self._config.key: obs_data}, {}

    def reset(self, ego: carla.Actor) -> None:
        # Spawn camera once — position updated in get_observation
        if self._camera is None:
            bp = self._world.get_blueprint("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self._config.shape[1]))
            bp.set_attribute("image_size_y", str(self._config.shape[0]))
            bp.set_attribute("fov", str(self._config.fov))
            bp.set_attribute("sensor_tick", str(self._config.sensor_tick))
            self._camera = self._world.spawn_unmanaged_actor(carla.Transform(), bp)
            self._camera.listen(self._update_data)

    def destroy(self) -> None:
        self._data = None
        if self._camera is not None:
            self._camera.destroy()
            self._camera = None

    def _update_data(self, data) -> None:
        camera_data = np.frombuffer(data.raw_data, dtype=np.uint8)
        camera_data = np.reshape(camera_data, (data.height, data.width, 4))
        camera_data = camera_data[:, :, :3]
        camera_data = camera_data[:, :, ::-1]
        self._data = camera_data
