import os
import sys

import numpy as np
import gymnasium as gym

from tdmpc2.envs.wrappers.time_limit import TimeLimit



class HumanoidWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "SLURM_STEP_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
        if "SLURM_JOB_GPUS" in os.environ:
            os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
            print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

        super().__init__(env)
        self.env = env
        self.cfg = cfg

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, done, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()


def make_env(cfg):
    """
    Make Humanoid environment.
    """
    import humanoid_bench

    env = gym.make(
        cfg.task,
    )
    env = HumanoidWrapper(env, cfg)
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps")
    return env
