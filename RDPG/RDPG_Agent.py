from collections import deque
import tensorflow as tf
from tensorflow_core.contrib import rnn
import numpy as np


class RDPG_Agent:
    def __init__(
            self,
            env,
            replay_size=80000,
            episode=10000,
            batch_size=32,
            gamma=0.99,
            step=1000,
            learning_rate=5e-4,
            update_freq=100
    ):
        # env params
        self.env = env
        self.state_dim = env.reset().shape[0]
        self.action_dim = env.action_space.shape[0]
        # replay buffer
        self.replay_buffer = deque()
        self.replay_size = replay_size
        # fit params
        self.EPISODE = episode
        self.STEP = step
        # params
        self.gamma = gamma
        self.batch_size = batch_size
        # training
        self.lr = learning_rate
        self.time_step = 0
        self.update_freq = update_freq
        # session
        self.sess = tf.Session()
        # actor net copy
        self.actor_assign_ops = []
        self.actor_params_dict = {}
        self.actor_holder_list = []
        # critic net copy
        self.critic_assign_ops = []
        self.critic_params_dict = {}
        self.critic_holder_list = []
        # create nets and training method
        self.create_actor_nets()
        self.create_critic_nets()
        self.training_method()
        self.sess.run(tf.global_variables_initializer())

    def create_actor_nets(self):
        self.actor_state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
