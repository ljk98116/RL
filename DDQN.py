from DQN import DQN
import random
import numpy as np

class DDQN(DQN):
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
        super(DDQN,self).__init__(
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

    def _training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)

        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]

        y_target = []
        q_eval = self.sess.run(self.q_eval,feed_dict={self.state:next_state_batch})
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})

        for i in range(self.batch_size):
            if done_batch[i]:
                y_target.append(reward_batch[i])
            else:
                idx = np.argmax(q_eval[i])
                y_target.append(reward_batch[i] + self.gamma * q_target[i][idx])
        
        # train_op
        self.sess.run(self._train_op,feed_dict={self.state:state_batch,self.y_target:y_target,self.action:action_batch})
