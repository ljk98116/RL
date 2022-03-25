from ..DQN import DQN
from collections import deque
import tensorflow as tf
import numpy as np

class Rainbow_DQN(DQN):
    def __init__(
        self,
        env,
        replay_size = 80000,
        EPISODE = 10000,
        batch_size = 32,
        gamma = 0.97,
        STEP = 1000,
        epsilon = 0.5,
        learning_rate = 1e-3,
        update_freq = 5
    ):
        self.params_cache = deque()
        super(Rainbow_DQN,self).__init__(
            env,
            replay_size,
            EPISODE,
            batch_size,
            gamma,
            STEP,
            epsilon,
            learning_rate,
            update_freq
        )    
        