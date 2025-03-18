from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class MultitaskTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		print('running evaluations...')
		task_idx = torch.zeros(1, dtype=torch.int32)[0] - 1
		task_rewards, task_successes = [], []
		for task in range(len(self.env.envs)):
			task_idx += 1
			ep_rewards, ep_successes = [], []
			for i in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx=task_idx.item()), False, 0, 0
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, truncated, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			task_rewards.append(np.nanmean(ep_rewards))
			task_successes.append(np.nanmean(ep_successes))
		results_eval = {}
		for idx in range(len(self.env.envs)):
			results_eval[f'episode_reward_{idx}'] = task_rewards[idx]
			results_eval[f'episode_success_{idx}'] = task_successes[idx]
		results_eval['episode_reward'] = np.nanmean(task_rewards)
		results_eval['episode_success'] = np.nanmean(task_successes)
		print('finished evaluations...')
		return results_eval

	def to_td(self, obs, action=None, reward=None, task=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
            task=task.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
        
		train_metrics, done, eval_next = {}, True, False
		task_idx = torch.zeros(1, dtype=torch.int32)[0] + len(self.env.envs)
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				task_idx += 1
				if task_idx >= len(self.env.envs):
					task_idx = torch.zeros(1, dtype=torch.int32)[0]
				obs = self.env.reset(task_idx=task_idx.item())
				self._tds = [self.to_td(obs=obs, task=task_idx)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1, task=task_idx)
			else:
				action = self.env.rand_act()
			obs, reward, done, truncated, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = 500
					print("Pretraining agent on seed data...")
					for _ in range(num_updates):
						_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)
				else:
					num_updates = 1
					if self._step % len(self.env.envs) == 0:
						for _ in range(num_updates):
							_train_metrics = self.agent.update(self.buffer)
						train_metrics.update(_train_metrics)


			self._step += 1

		self.logger.finish(self.agent)
