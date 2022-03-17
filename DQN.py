import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQN:
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

        self.replay_buffer = deque()
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

    def create_nets(self):
        self.state = tf.placeholder(tf.float32,[None,self.state_dim],name='state')
        self.action = tf.placeholder(tf.int32,[None],name='action')
        self.next_state = tf.placeholder(tf.float32,[None,self.state_dim],name='next_state')
        
        w_initializer,b_initializer = tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)
        with tf.variable_scope('online_net'):
            e1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            # get Q(state)
            self.q_eval = tf.layers.dense(e1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer)

        # get Q for each action,Q(state,action)
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0],dtype=tf.int32),self.action],axis=1)
        self.q_eval_w = tf.gather_nd(params=self.q_eval,indices=a_indices)

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t1')
            # get Q(state)
            self.q_target = tf.layers.dense(t1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer)

        # network copy ops
        online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='online_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')

        for var in online_params:
            self.holder_list.append(tf.placeholder(tf.float32,[1]+var.shape.as_list()))
            self.params_dict[self.holder_list[len(self.holder_list)-1]] = np.zeros(var.shape.as_list())
        
        for i,var in zip(range(len(target_params)),target_params):
            self.assign_ops.append(tf.assign(var,self.holder_list[i][0]))

        self.update_net_op = tf.group(*self.assign_ops)            

    def training_method(self):
        with tf.variable_scope('training'):
            self.y_target = tf.placeholder(tf.float32,[None],name='y_target')
            self.loss = tf.reduce_mean(tf.squared_difference(self.y_target,self.q_eval_w,name='TD_error'))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
    def choose_action(self,state):
        state = state[np.newaxis,:]

        if np.random.uniform() > self.epsilon:
            action = self.sess.run(self.q_eval,feed_dict={self.state:state})
            res = np.argmax(action)
        else:
            res = np.random.randint(0,self.action_dim)

        self.epsilon -= 0.001
        return res

    def action_direct(self,state):
        state = state[np.newaxis,:]
        action = self.sess.run(self.q_eval,feed_dict={self.state:state})
        res = np.argmax(action)
        return res
    
    def store_transition(self,state,action,reward,next_state,done):
        if len(self.replay_buffer) >= self.replay_size:
            self.replay_buffer.popleft()
        self.replay_buffer.append((state,action,reward,next_state,done))
    
    def update_net(self):
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
        
        self.sess.run(self.update_net_op,feed_dict=self.params_dict)
        # print('target2',self.sess.run(target_params))

    def training(self):
        if len(self.replay_buffer) > self.batch_size:
            self._training()
        
        if self.time_step % self.update_freq == 0:
            self.update_net()
    
    def _training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)

        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]

        y_target = []
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state_batch})

        for i in range(self.batch_size):
            if done_batch[i]:
                y_target.append(reward_batch[i])
            else:
                y_target.append(reward_batch[i] + self.gamma * np.max(q_target[i]))
        
        # train_op
        self.sess.run(self._train_op,feed_dict={self.state:state_batch,self.y_target:y_target,self.action:action_batch})

    def evaluate(self,ep):
        score = 0
        for j in range(10):
            total_reward = 0
            state = self.env.reset()
            for step in range(self.STEP):
                self.env.render()
                action = self.action_direct(state)
                next_state,reward,done,info = self.env.step(action)
                total_reward += reward
                if done:
                    break
                state = next_state
            score += total_reward

        print('Episode:%d,Reward:%f'%(ep,score / 10.0))
        if score / 10.0 >= 500:
            return True
        return False
    
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

            if ep % 10 == 0:
                if self.evaluate(ep):
                    print('You Win !!')
                    break