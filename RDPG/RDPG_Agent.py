import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow_core.contrib import rnn


class RDPG_Agent:
    def __init__(
            self,
            env,
            replay_size=80000,
            episode=10000,
            batch_size=64,
            gamma=0.99,
            step=200,
            T = 5,
            hidden_units = 200,
            learning_rate=1e-4,
            update_freq=50
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
        # create nets and training method
        self.T = T
        self.hidden_units = hidden_units

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

        # nets and assign ops
        self.Actor()
        self.TargetActor()

        self.Critic()
        self.TargetCritic()

        self.softupdate()

        self.training_method()

        # 初始化所有变量
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def length(data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def Actor(self):
        # batch_size x T x state_dim
        self.kernel_initializer = tf.random_normal_initializer(0., 0.3)
        self.bias_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope('actor_net'):
            # forward process
            self.actor_state = tf.placeholder(tf.float32, [None, self.T, self.state_dim], name='state')

            self.actor_lstm = rnn.LSTMCell(self.hidden_units)
            self.actor_init_state = self.actor_lstm.zero_state(self.batch_size, tf.float32)
            self.actor_lstm_outputs, self.actor_final_state = \
                tf.nn.dynamic_rnn(self.actor_lstm, self.actor_state,
                                  sequence_length=self.length(self.actor_state),
                                  initial_state=self.actor_init_state, dtype=tf.float32)
            # T x batch_size x num_units
            self.actor_lstm_outputs_list = \
                tf.reshape(self.actor_lstm_outputs, [-1, self.hidden_units])

            actor_e1 = tf.layers.dense(
                self.actor_lstm_outputs_list, 200, activation='relu',
                kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)

            self.actor_output = tf.layers.dense(
                actor_e1, self.action_dim,
                kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            )

            self.actor_output = tf.reshape(self.actor_output, [-1, self.T, self.action_dim])
            '''
            # 离散情况转换为softmax分布后进行采样
            self.actor_output = tf.nn.softmax(self.actor_output)
            self.actor_output = tf.distributions.Categorical(self.actor_output).sample()
            '''
            self.actor_act = tf.reshape(self.actor_output, [-1,])

    def TargetActor(self):
        # batch_size x T x state_dim
        with tf.variable_scope('target_actor_net'):
            # forward process
            self.t_actor_state = tf.placeholder(tf.float32, [None, self.T, self.state_dim], name='state')

            self.t_actor_lstm = rnn.LSTMCell(self.hidden_units)
            self.t_actor_init_state = self.t_actor_lstm.zero_state(self.batch_size, tf.float32)
            self.t_actor_lstm_outputs, self.t_actor_final_state = \
                tf.nn.dynamic_rnn(self.t_actor_lstm, self.t_actor_state,
                                  sequence_length=self.length(self.t_actor_state),
                                  initial_state=self.t_actor_init_state, dtype=tf.float32)
            # T x batch_size x num_units
            self.t_actor_lstm_outputs_list = \
                tf.reshape(self.t_actor_lstm_outputs, [-1, self.hidden_units])

            t_actor_e1 = tf.layers.dense(
                self.t_actor_lstm_outputs_list, 200, activation='relu',
                kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)

            self.t_actor_output = tf.layers.dense(
                t_actor_e1, self.action_dim,
                kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            )

            self.t_actor_output = tf.reshape(self.t_actor_output, [-1, self.T, self.action_dim])
            '''
            # 离散情况转换为softmax分布后进行采样
            self.t_actor_output = tf.nn.softmax(self.t_actor_output)
            self.t_actor_output = tf.distributions.Categorical(self.t_actor_output).sample()
            '''
            self.t_actor_act = tf.reshape(self.t_actor_output, [-1, ])

    def Critic(self):
        with tf.variable_scope('critic_net'):
            self.critic_state = tf.placeholder(tf.float32,[None,self.T,self.state_dim],name='critic_state')
            self.critic_action = tf.placeholder(tf.float32,[None,self.T,self.action_dim],name='critic_action')

            # 将state和action在维度上合并
            critic_state = tf.reshape(self.critic_state,[-1,self.state_dim])
            critic_action = tf.reshape(self.critic_action,[-1,self.action_dim])

            self.critic_input = tf.concat([critic_state, critic_action], axis=1)

            e1 = tf.layers.dense(
                self.critic_input,200,activation='relu',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )

            e2 = tf.layers.dense(
                e1,200,activation='relu',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            self.critic_output = tf.layers.dense(
                e2, 1,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            self.critic_output = tf.reshape(self.critic_output,[-1,self.T])

    def TargetCritic(self):
        with tf.variable_scope('target_critic_net'):
            self.t_critic_state = tf.placeholder(tf.float32, [None, self.T, self.state_dim], name='target_critic_state')
            self.t_critic_action = \
                tf.placeholder(tf.float32, [None, self.T, self.action_dim], name='target_critic_action')

            # 将state和action在维度上合并
            critic_state = tf.reshape(self.t_critic_state, [-1,self.state_dim])
            critic_action = tf.reshape(self.t_critic_action, [-1,self.action_dim])

            self.t_critic_input = tf.concat([critic_state, critic_action], axis=1)
            e1 = tf.layers.dense(
                self.t_critic_input,200,activation='relu',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            e2 = tf.layers.dense(
                e1,200,activation='relu',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            self.t_critic_output = tf.layers.dense(
                e2, 1,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            self.t_critic_output = tf.reshape(self.t_critic_output,[-1,self.T])

    def training_method(self):
        # Actor训练过程
        # batch_size x T
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor_net')

        self.critic_y_target = tf.placeholder(tf.float32, [None,self.T], name='critic_y_target')
        self.critic_loss = tf.losses.mean_squared_error(self.critic_output, self.critic_y_target)
        # critic网络的梯度
        self.critic_grad = tf.gradients(self.critic_loss, critic_variables)
        # critic关于输入critic的动作的梯度,需要critic_y_target,critic_state,critic_action(由actor网络输出)
        self.critic_grad_ = tf.gradients(self.critic_loss, self.critic_action)
        # 训练critic网络,需要critic_y_target,critic_state,critic_action(由replay_buffer抽样给出)
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.critic_train_op = self.critic_optimizer.apply_gradients(zip(self.critic_grad, critic_variables))
        # 训练actor网络
        self.critic_grad_input = tf.placeholder(tf.float32, [None, self.T, self.action_dim], name='critic_grad_input')
        self.actor_grad = tf.gradients(self.actor_output, actor_variables, self.critic_grad_input)
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.lr)
        self.actor_train_op = self.actor_optimizer.apply_gradients(zip(self.actor_grad,actor_variables))

    def softupdate(self):
        # actor network copy ops
        actor_online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        actor_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor_net')

        for var in actor_online_params:
            self.actor_holder_list.append(tf.placeholder(tf.float32, [1] + var.shape.as_list()))
            self.actor_params_dict[self.actor_holder_list[len(self.actor_holder_list) - 1]] = np.zeros(var.shape.as_list())

        for i, var in zip(range(len(actor_target_params)), actor_target_params):
            self.actor_assign_ops.append(tf.assign(var, self.actor_holder_list[i][0]))

        self.actor_update_net_op = tf.group(*self.actor_assign_ops)

        # critic network copy ops
        critic_online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        critic_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_net')

        for var in critic_online_params:
            self.critic_holder_list.append(tf.placeholder(tf.float32, [1] + var.shape.as_list()))
            self.critic_params_dict[self.critic_holder_list[len(self.critic_holder_list) - 1]] = np.zeros(var.shape.as_list())

        for i, var in zip(range(len(critic_target_params)), critic_target_params):
            self.critic_assign_ops.append(tf.assign(var, self.critic_holder_list[i][0]))

        self.critic_update_net_op = tf.group(*self.critic_assign_ops)

    def update_net(self):
        actor_online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')
        for item, e in zip(self.actor_holder_list, actor_online_params):
            value = self.sess.run(e)
            value = value[np.newaxis, :]
            self.actor_params_dict[item] = value

        self.sess.run(self.actor_update_net_op, feed_dict=self.actor_params_dict)

        critic_online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        for item, e in zip(self.critic_holder_list, critic_online_params):
            value = self.sess.run(e)
            value = value[np.newaxis, :]
            self.critic_params_dict[item] = value

        self.sess.run(self.critic_update_net_op, feed_dict=self.critic_params_dict)

    def choose_action(self, state):
        state_z = np.zeros((self.batch_size, self.T, self.state_dim))
        state_z[0][self.time_step % self.T] = state
        action = self.sess.run(self.actor_output, feed_dict={self.actor_state:state_z})[0][self.time_step % self.T]
        return action

    def store_transition(self,state,action,reward,next_state,done):
        if len(self.replay_buffer) == self.replay_size:
            self.replay_buffer.popleft()
        self.replay_buffer.append([state,action,reward,next_state,done])

    def training(self):
        if len(self.replay_buffer) > self.batch_size * self.T:
            self._training()

        if self.time_step % self.update_freq == 0:
            self.update_net()

    # batch_size x T
    def sample(self):
        size = len(self.replay_buffer)
        max_num = size - self.T
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        for i in range(self.batch_size):
            idx = random.randint(0,max_num)
            state_T = []
            action_T = []
            reward_T = []
            next_state_T = []
            done_T = []
            for j in range(idx,idx + self.T):
                state_T.append(self.replay_buffer[j][0])
                action_T.append(self.replay_buffer[j][1])
                reward_T.append(self.replay_buffer[j][2])
                next_state_T.append(self.replay_buffer[j][3])
                done_T.append(self.replay_buffer[j][4])
            state.append(state_T)
            action.append(action_T)
            reward.append(reward_T)
            next_state.append(next_state_T)
            done.append(done_T)
        return state,action,reward,next_state,done

    def _training(self):
        state,action,reward,next_state,done = self.sample()
        action_target = self.sess.run(self.t_actor_output,feed_dict={self.t_actor_state:next_state})
        Q_target = self.sess.run(
            self.t_critic_output,
            feed_dict={self.t_critic_state:next_state, self.t_critic_action:action_target}
        )

        y_target = []
        for i in range(self.batch_size):
            y_target_T = []
            for j in range(self.T):
                y_target_T.append(reward[i][j] + self.gamma * (1 - done[i][j]) * Q_target[i][j])
            y_target.append(y_target_T)
        # 获取critic关于action的梯度
        action_eval = self.sess.run(self.actor_output,feed_dict={self.actor_state:state})
        critic_action_grad = self.sess.run(
            self.critic_grad_,
            feed_dict={
                self.critic_y_target:y_target,
                self.critic_state:state,
                self.critic_action:action_eval
            }
        )[0]

        # 训练critic网络
        self.sess.run(
            self.critic_train_op,
            feed_dict={
                self.critic_state:state,
                self.critic_y_target:y_target,
                self.critic_action:action
            }
        )

        # 训练actor网络
        self.sess.run(
            self.actor_train_op,
            feed_dict={
                self.actor_state:state,
                self.critic_grad_input:critic_action_grad
            }
        )

    def evaluate(self, ep):
        score = 0
        for j in range(10):
            total_reward = 0
            state = self.env.reset()
            for step in range(self.STEP):
                self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
                state = next_state
            score += total_reward

        print('Episode:%d,Reward:%f' % (ep, score / 10.0))

    def fit(self):
        for ep in range(self.EPISODE):
            state = self.env.reset()
            self.time_step = 0
            for step in range(self.STEP):
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                # reward = -1 if done else 0.1
                self.store_transition(state, action, reward, next_state, done)
                if self.time_step % self.T == 0:
                    self.training()
                if done:
                    break
                state = next_state
                self.time_step += 1

            if ep % 50 == 0:
                self.evaluate(ep)


