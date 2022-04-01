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
        learning_rate=1e-3, 
        update_freq=100,
        # n step replay
        n = 10
    ):
        self.n = n
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
    
    def training(self):
        if len(self.replay_buffer) >= self.batch_size + self.n:
            self._training()
        
        if self.time_step % self.update_freq == 0:
            self.update_net()    

    def n_step_sample(self):
        rr = len(self.replay_buffer) - self.n
        random_index = random.sample(range(0,rr),self.batch_size)
        batch = []
        for i in range(self.batch_size):
            batch.append(self.replay_buffer[random_index[i]])
        return batch,random_index

    def _training(self):
        batch,indexes = self.n_step_sample()

        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]

        y_target = []
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})

        for i in range(self.batch_size):
            # calculate n step reward
            n_step_reward = 0.
            for j in range(self.n):
                n_step_reward += self.gamma ** j * self.replay_buffer[indexes[i] + j][2]
            
            if done_batch[i]:
                y_target.append(n_step_reward)
            else:
                y_target.append(n_step_reward + self.gamma ** self.n * np.max(q_target[i]))
        
        # train_op
        self.sess.run(self._train_op,feed_dict={self.state:state_batch,self.y_target:y_target,self.action:action_batch})    
