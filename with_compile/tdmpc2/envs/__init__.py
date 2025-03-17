from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper



def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

from envs.humanoid import make_env as make_humanoid_env

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		env = None
		for fn in [make_humanoid_env]:
			try:
				env = fn(cfg)
			except ValueError:
				pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env


'''
import torch

class config:
    
    tasks=['h1hand-crawl-v0']
    multitask=True
    
cfg = config()
env = make_env(cfg)
env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminal, truncate, info = env.step(torch.from_numpy(action))



'''