from tdmpc2.common.parser import parse_cfg
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.trainer.online_trainer import OnlineTrainer
from tdmpc2.common.buffer import Buffer
from tdmpc2.common.logger import Logger
import torch
import omegaconf
import numpy as np


cfg = omegaconf.OmegaConf.load('/Users/mnauman/Documents/GitHub/tdmpc2/tdmpc2/tdmpc2/config.yaml')
cfg = parse_cfg(cfg)


trainer_cls = OnlineTrainer
trainer = trainer_cls(
    cfg=cfg,
    env=make_env(cfg),
    agent=TDMPC2(cfg),
    buffer=Buffer(cfg),
    logger=Logger(cfg),
)
#trainer.train()


train_metrics, done, eval_next = {}, True, True
task_idx = torch.zeros(1, dtype=torch.int32)[0] + len(trainer.env.envs)


while trainer._step <= trainer.cfg.steps:
    # Evaluate agent periodically
    if trainer._step % trainer.cfg.eval_freq == 0:
        eval_next = True

    # Reset environment
    if done:
        if trainer._step > 0:
            trainer._ep_idx = trainer.buffer.add(torch.cat(trainer._tds))
        task_idx += 1
        if task_idx >= len(trainer.env.envs):
            task_idx = torch.zeros(1, dtype=torch.int32)[0]
        obs = trainer.env.reset(task_idx=task_idx.item())[0]
        trainer._tds = [trainer.to_td(obs=obs, task=task_idx)]
        _ = trainer.agent.act(obs, t0=len(trainer._tds) == 1, task=task_idx)

    # Collect experience
    if trainer._step > trainer.cfg.seed_steps:
        action = trainer.agent.act(obs, t0=len(trainer._tds) == 1, task=task_idx)
    else:
        action = trainer.env.rand_act()
    obs, reward, done, truncated, info = trainer.env.step(action)
    done = done or truncated
    trainer._tds.append(trainer.to_td(obs=obs, action=action, reward=reward, task=task_idx))
    trainer._step += 1
    
    # Update agent
    if trainer._step >= trainer.cfg.seed_steps:
        if trainer._step == trainer.cfg.seed_steps:
            num_updates = 500
            print("Pretraining agent on seed data...")
            for _ in range(num_updates):
                _train_metrics = trainer.agent.update(trainer.buffer)
            train_metrics.update(_train_metrics)
        else:
            num_updates = 1
            if trainer._step % 9 == 0:                        
                for _ in range(num_updates):
                    _train_metrics = trainer.agent.update(trainer.buffer)
                train_metrics.update(_train_metrics)







from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

trainer.buffer.save_buffer('test')
trainer.buffer.load_buffer('test', cfg)


with open(f"test/num_eps", "r") as file:
    init_values = file.read()


_storage = LazyTensorStorage(trainer.buffer._capacity, device=torch.device('cpu'))
_sampler = SliceSampler(num_slices=trainer.cfg.batch_size, end_key=None, traj_key="episode", truncated_key=None)
trainer.buffer._buffer = ReplayBuffer(
    storage=storage,
    sampler=_sampler,
    #pin_memory=True,
    prefetch=0,
    batch_size=trainer.buffer._batch_size,
)

trainer.buffer._buffer.loads("test.txt")

len(trainer.buffer._buffer)
trainer.buffer.num_eps

obs, action, reward, task = trainer.buffer.sample()
