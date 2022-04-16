from DQN import DQN
import random
import numpy as np
import tensorflow as tf
from collections import deque
from priority_replay_buffer import priority_replay_buffer

class priority_DQN(DQN):
    def __init__(
        self,
        env,
        replay_size = 20000,
        EPISODE = 100000,
        batch_size = 32,
        gamma = 0.97,
        STEP = 1000,
        epsilon = 0.5,
        learning_rate = 1e-3,
        learning_rate_decay = 0.975,
        update_freq = 200,
        alpha = 0.6,
        beta = 0.4,
        beta_step = 0.0001,
        e = 0.0005
    ):
        self.env = env

        self.replay_buffer = priority_replay_buffer(replay_size)
        self.replay_size = replay_size

        self.EPISODE = EPISODE
        self.STEP = STEP

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.lr = learning_rate
        self.lr_decay = learning_rate_decay

        self.alpha = alpha
        self.beta = beta
        self.beta_step = beta_step
        self.e = e

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

    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.store_priority(state,action,reward,next_state,done)

    def training_method(self):
        with tf.variable_scope('training'):
            self.y_target = tf.placeholder(tf.float32,[None],name='y_target')
            self.weight = tf.placeholder(tf.float32,[None],name='weight')

            self.loss = tf.reduce_mean(tf.multiply(self.weight,tf.squared_difference(self.y_target,self.q_eval_w,name='TD_error')))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def training(self):
        # print(len(self.replay_buffer.replay_buffer))
        if self.replay_buffer.p_buffer.total_times >= self.replay_size:
            # print("OK")
            self._training()
        
        if self.time_step % self.update_freq == 0:
            self.update_net()
    
    def _training(self):
        batch,idx,priority = self.replay_buffer.sample(self.batch_size)
        # print(batch)
        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]

        y_target = []
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})
        q_eval = self.sess.run(self.q_eval,feed_dict={self.state:state_batch})
        weight = []

        for i in range(self.batch_size):
            # calculate wi
            prob = (priority[i] / (self.replay_buffer.p_buffer.tree[0] + 0.)) ** self.alpha
            w = min(1.,(1 / (len(self.replay_buffer.p_buffer.leaf) * prob * 1.)) ** self.beta)
            weight.append(w)

            iidx = np.argmax(q_eval[i])
            if done_batch[i]:
                y_target.append(reward_batch[i])
            else:
                y_target.append(reward_batch[i] + self.gamma * q_target[i][iidx])

        # train_op
        self.sess.run(self._train_op,
        feed_dict={
            self.state:state_batch,
            self.y_target:y_target,
            self.action:action_batch,
            self.weight:weight
            }
        )

        # update priority
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})
        q_eval = self.sess.run(self.q_eval,feed_dict={self.state:state_batch})
        # print(self.replay_buffer.p_buffer.leaf[0])

        for i in range(self.batch_size):
            iidx = np.argmax(q_eval[i])
            # update priority
            td_error = np.abs(reward_batch[i] + self.gamma * q_target[i][iidx] * (1-done_batch[i]) - q_eval[i][action_batch[i]]) + self.e
            self.replay_buffer.modify(td_error,idx[i])

        self.beta = min(1.,self.beta + self.beta_step + 0.)

    def fit(self):
        for ep in range(self.EPISODE):
            state = self.env.reset()
            self.time_step = 0
            for step in range(self.STEP):
                action = self.choose_action(state)
                next_state,reward,done,info = self.env.step(action)
                reward = -1 if done else 0.1
                self.store_transition(state,action,reward,next_state,done)
                self.training()
                if done:
                    break
                state = next_state
                self.time_step += 1

            if ep % 100 == 0:
                if self.evaluate(ep):
                    print('You Win !!')
                    break