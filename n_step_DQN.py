import tensorflow as tf
import random
import numpy as np
from collections import deque
from DQN import DQN

class n_step_DQN(DQN):
    def __init__(
        self, 
        env, 
        replay_size=80000, 
        EPISODE=10000, 
        batch_size=32, 
        gamma=0.99, 
        STEP=1000, 
        epsilon=0.5, 
        learning_rate=0.0005, 
        update_freq=100
    ):
        super(n_step_DQN,self).__init__(
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
        