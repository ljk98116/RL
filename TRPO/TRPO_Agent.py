'''
TRPO思路：让策略分布尽可能接近的情况下，使得策略改进后的优势尽可能大
'''
import scipy.optimize
import tensorflow as tf
import numpy as np
from collections import deque

class TRPO_Agent:
    def __init__(
            self,
            env,
            replay_size = 10000,
            episode = 300,
            batch_size = 32,
            gamma = 0.9,
            tau = 0.6,
            step = 200,
            learning_rate = 1e-3,
            update_freq = 10
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
        self.tau = tau
        self.batch_size = batch_size
        # training
        self.lr = learning_rate
        self.time_step = 0
        self.update_freq = update_freq
        # critic net copy
        self.critic_assign_ops = []
        self.critic_params_dict = {}
        self.critic_holder_list = []

        # actor net copy
        self.actor_assign_ops = []
        self.actor_params_dict = {}
        self.actor_holder_list = []

        # session
        self.sess = tf.Session()
        self.state = tf.placeholder(tf.float32,[None,self.state_dim])
        self.t_state = tf.placeholder(tf.float32,[None,self.state_dim])

        # nets and assign ops
        self.Actor()
        self.TargetActor()

        self.Critic()
        self.TargetCritic()

        self.softupdate()

        self.training_method()

        # 初始化所有变量
        self.sess.run(tf.global_variables_initializer())

        self.var = 3
        self.render = 0

    @staticmethod
    def length(data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def Actor(self):
        # batch_size x state_dim
        with tf.variable_scope('actor_net'):
            # forward process
            self.actor_state = self.state

            actor_e1 = tf.layers.dense(
                self.actor_state, 30, activation='relu')

            self.actor_output = tf.layers.dense(
                actor_e1, self.action_dim, activation='tanh'
            )

            self.actor_output = tf.multiply(self.actor_output,self.action_dim)
            '''
            # 离散情况转换为softmax分布后进行采样
            self.actor_output = tf.nn.softmax(self.actor_output)
            self.actor_output = tf.distributions.Categorical(self.actor_output).sample()
            '''

    def TargetActor(self):
        # batch_size x state_dim
        with tf.variable_scope('target_actor_net'):
            # forward process
            self.t_actor_state = self.t_state

            t_actor_e1 = tf.layers.dense(
                self.t_actor_state, 30, activation='relu')

            self.t_actor_output = tf.layers.dense(
                t_actor_e1, self.action_dim, activation='tanh'
            )

            '''
            # 离散情况转换为softmax分布后进行采样
            self.t_actor_output = tf.nn.softmax(self.t_actor_output)
            self.t_actor_output = tf.distributions.Categorical(self.t_actor_output).sample()
            '''

    def Critic(self):
        with tf.variable_scope('critic_net'):
            self.critic_state = self.state
            self.critic_action = self.actor_output

            # 将state和action在维度上合并
            critic_state = tf.reshape(self.critic_state,[-1,self.state_dim])
            critic_action = tf.reshape(self.critic_action,[-1,self.action_dim])

            self.critic_input = tf.concat([critic_state, critic_action], axis=1)

            e1 = tf.layers.dense(
                self.critic_input,30,activation='relu'
            )

            e2 = tf.layers.dense(
                e1,30,activation='relu',
            )
            self.critic_output = tf.layers.dense(
                e2, 1,
            )

    def TargetCritic(self):
        with tf.variable_scope('target_critic_net'):
            self.t_critic_state = self.t_state
            self.t_critic_action = self.t_actor_output

            # 将state和action在维度上合并
            critic_state = tf.reshape(self.t_critic_state, [-1,self.state_dim])
            critic_action = tf.reshape(self.t_critic_action, [-1,self.action_dim])

            self.t_critic_input = tf.concat([critic_state, critic_action], axis=1)
            e1 = tf.layers.dense(
                self.t_critic_input,30,activation='relu',
            )
            e2 = tf.layers.dense(
                e1,30,activation='relu',
            )
            self.t_critic_output = tf.layers.dense(
                e2, 1,
            )

    def training_method(self):
        critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net')
        actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='actor_net')

        self.critic_y_target = tf.placeholder(tf.float32, [None], name='critic_y_target')
        self.critic_loss = tf.losses.mean_squared_error( self.critic_y_target,self.critic_output)
        # 训练critic网络,需要critic_y_target,critic_state,critic_action(由replay_buffer抽样给出)
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss,var_list=critic_variables)
        # 训练actor网络
        self.actor_loss = -tf.reduce_mean(self.critic_output)
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.actor_train_op = self.actor_optimizer.minimize(self.actor_loss,var_list=actor_variables)

    def softupdate(self):
        # actor network copy ops
        actor_online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_net')
        actor_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor_net')

        for var in actor_online_params:
            self.actor_holder_list.append(tf.placeholder(tf.float32, [1] + var.shape.as_list()))
            self.actor_params_dict[self.actor_holder_list[len(self.actor_holder_list) - 1]] = np.zeros(var.shape.as_list())

        for i, var in zip(range(len(actor_target_params)), actor_target_params):
            self.actor_assign_ops.append(tf.assign(var, self.actor_holder_list[i][0]))

        self.actor_update_net_op = tf.group(*self.actor_assign_ops)

        # critic network copy ops
        critic_online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net')
        critic_target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic_net')

        for var in critic_online_params:
            self.critic_holder_list.append(tf.placeholder(tf.float32, [1] + var.shape.as_list()))
            self.critic_params_dict[self.critic_holder_list[len(self.critic_holder_list) - 1]] = np.zeros(var.shape.as_list())

        for i, var in zip(range(len(critic_target_params)), critic_target_params):
            self.critic_assign_ops.append(tf.assign(var, self.critic_holder_list[i][0]))

        self.critic_update_net_op = tf.group(*self.critic_assign_ops)

    def update_net(self):
        actor_online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_net')
        for item, e in zip(self.actor_holder_list, actor_online_params):
            value = self.sess.run(e)
            value = value[np.newaxis, :]
            self.actor_params_dict[item] = value

        self.sess.run(self.actor_update_net_op, feed_dict=self.actor_params_dict)

        critic_online_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net')
        for item, e in zip(self.critic_holder_list, critic_online_params):
            value = self.sess.run(e)
            value = value[np.newaxis, :]
            self.critic_params_dict[item] = value

        self.sess.run(self.critic_update_net_op, feed_dict=self.critic_params_dict)