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
trainer.train()


