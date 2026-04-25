import gymnasium
import numpy as np


class ThirdPersonRenderWrapper(gymnasium.Wrapper):
    """Expose a synchronized third-person render through the info dict."""

    def __init__(self, env, camera='front_pixels', info_key='third_person_observation'):
        super().__init__(env)
        self.camera = camera
        self.info_key = info_key

    def _get_third_person_observation(self):
        frame = self.unwrapped.render(camera=self.camera)
        return np.array(frame, copy=True)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        info = dict(info)
        info[self.info_key] = self._get_third_person_observation()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[self.info_key] = self._get_third_person_observation()
        return observation, reward, terminated, truncated, info
