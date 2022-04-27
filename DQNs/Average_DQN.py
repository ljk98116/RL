from os import name
from threading import currentThread
from DQN import DQN
import random
import numpy as np
import tensorflow as tf
from collections import deque
import time

class Average_DQN(DQN):
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
        update_freq = 5,
        K = 10
    ):
        self.K = K
        self.params_cache = deque()
        super(Average_DQN,self).__init__(
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

    def update_net(self):
        if len(self.params_cache) >= self.K:
            self.params_cache.popleft()
        online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='online_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')
        # print('online',self.sess.run(online_params))
        # print('target1',self.sess.run(target_params))
        '''
        for t,e in zip(target_params,online_params):
            assign_op = tf.assign(t,e)
            self.sess.run(assign_op)
        '''

        for item,e in zip(self.holder_list,online_params):
            value = self.sess.run(e)
            value = value[np.newaxis,:]
            self.params_dict[item] = value
        
        self.params_cache.append(self.sess.run(target_params))
        # print(self.params_cache[-1])
        self.sess.run(self.update_net_op,feed_dict=self.params_dict)
        # print('target2',self.sess.run(target_params))

    def _training(self):
        # print(self.time_step)
        batch = random.sample(self.replay_buffer,self.batch_size)

        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]

        y_target = []

        final_q_target = np.zeros((self.batch_size,self.action_dim))

        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')
        current_params = self.sess.run(target_params)

        # print(self.params_cache)
        for j in range(len(self.params_cache)):
            # print(j,self.params_cache[j])
            # update target net
            for item,e in zip(self.holder_list,self.params_cache[j]):
                value = e[np.newaxis,:]
                self.params_dict[item] = value
            self.sess.run(self.update_net_op,feed_dict=self.params_dict)

            # target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')
            # print(self.sess.run(target_params))

            # add to final
            final_q_target += self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})

        # recover
        for item,e in zip(self.holder_list,current_params):
            value = e[np.newaxis,:]
            self.params_dict[item] = value
        self.sess.run(self.update_net_op,feed_dict=self.params_dict)

        for i in range(self.batch_size):
            if done_batch[i]:
                y_target.append(reward_batch[i])
            else:
                y_target.append(reward_batch[i] + self.gamma * np.max(final_q_target[i]) / len(self.params_cache))
           
        # train_op
        self.sess.run(self._train_op,feed_dict={self.state:state_batch,self.y_target:y_target,self.action:action_batch})
