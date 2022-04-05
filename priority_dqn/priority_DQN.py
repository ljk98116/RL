from ..DQN import DQN
import random
import numpy as np
import tensorflow as tf
from collections import deque
from priority_replay_buffer import priority_replay_buffer
class priority_DQN(DQN):
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
        self.env = env

        self.replay_buffer = priority_replay_buffer()
        self.replay_size = replay_size

        self.EPISODE = EPISODE
        self.STEP = STEP

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.lr = learning_rate

        self.state_dim = self.env.reset().shape[0]
        self.action_dim = self.env.action_space.n

        self.time_step = 0
        self.update_freq = update_freq

        # session
        self.sess = tf.Session()

        self.assign_ops = []
        self.params_dict = {}
        self.holder_list = []

        # neoral net and training method
        self.create_nets()
        self.training_method()
        self.sess.run(tf.global_variables_initializer())
    