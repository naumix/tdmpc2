from tdmpc2.common.parser import parse_cfg
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.trainer.online_trainer import OnlineTrainer
from tdmpc2.common.buffer import Buffer
from tdmpc2.common.logger import Logger
import torch
import omegaconf
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


task_idx = torch.zeros((1), dtype=torch.int32)
obs = trainer.env.reset(task_idx=task_idx.item())[0]
trainer._tds = [trainer.to_td(obs=obs, task=task_idx)]
# Collect experience
running = 0

while running < 10:
    for i in range(1000):
        action = trainer.agent.act(obs, t0=len(trainer._tds) == 1, task=trainer.env.task_idx)
        obs, reward, done, truncated, info = trainer.env.step(action)
        print(i, done, truncated)
        done = done or truncated
        trainer._tds.append(trainer.to_td(obs=obs, action=action, reward=reward, task=task_idx))
        if done:
            #running += 100
            #break
            a = trainer._tds
            len(a)
            episode = torch.cat(trainer._tds)
            trainer._ep_idx = trainer.buffer.add(episode)
            task_idx += 1
            if task_idx == len(trainer.env.envs):
                task_idx = 0
            obs = trainer.env.reset(task_idx=task_idx.item())[0]
            trainer._tds = [trainer.to_td(obs)]
            running += 1
            
aa = a[0]
ab = a[-1]
batch = trainer.buffer.sample()
_train_metrics = trainer.agent.update(trainer.buffer)


episode['obs']
