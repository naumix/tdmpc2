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
        if eval_next:
            eval_metrics = trainer.eval()
            eval_metrics.update(trainer.common_metrics())
            trainer.logger.log(eval_metrics, "eval")
            eval_next = False

        if trainer._step > 0:
            train_metrics.update(
                episode_reward=torch.tensor(
                    [td["reward"] for td in trainer._tds[1:]]
                ).sum(),
                episode_success=info["success"],
            )
            train_metrics.update(trainer.common_metrics())

            results_metrics = {
                               'return': train_metrics['episode_reward'],
                               'episode_length': len(trainer._tds[1:]),
                               'success': train_metrics['episode_success'],
                               'success_subtasks': info['success_subtasks'],
                               'step': trainer._step,}
            
            for task_index in range(len(trainer.env.envs)):
                results_metrics[f'return_{task_index}'] = train_metrics[f'episode_reward_{task_index}']
                results_metrics[f'success_{task_index}'] = train_metrics[f'episode_success_{task_index}']
        
            trainer.logger.log(train_metrics, "train")
            trainer.logger.log(results_metrics, "results")
            trainer._ep_idx = trainer.buffer.add(torch.cat(trainer._tds))
        

        obs = trainer.env.reset()[0]
        trainer._tds = [trainer.to_td(obs=obs, task=task_idx)]
        task_idx += 1
        if task_idx >= len(trainer.env.envs):
            task_idx = torch.zeros(1, dtype=torch.int32)[0]

    print(task_idx)

    # Collect experience
    if trainer._step > trainer.cfg.seed_steps:
        action = trainer.agent.act(obs, t0=len(trainer._tds) == 1)
    else:
        action = trainer.env.rand_act()
    obs, reward, done, truncated, info = trainer.env.step(action)
    done = done or truncated
    trainer._tds.append(trainer.to_td(obs, action, reward))
    
    if trainer._step > trainer.cfg.seed_steps:
        break
        _train_metrics = trainer.agent.update(trainer.buffer)
        train_metrics.update(_train_metrics)
