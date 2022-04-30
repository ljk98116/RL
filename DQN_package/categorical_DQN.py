import tensorflow as tf
import random
import numpy as np
from collections import deque
from DQN import DQN

class categorical_DQN(DQN):
    def __init__(
        self,
        env,
        replay_size = 80000,
        EPISODE = 10000,
        batch_size = 32,
        gamma = 0.99,
        STEP = 1000,
        epsilon = 0.5,
        learning_rate = 1e-3,
        update_freq = 100,
        Vmin = -10,
        Vmax = 10,
        N_atoms = 51
    ):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.N_atoms = N_atoms
        # initialize support vector
        self.z = np.linspace(self.Vmin,self.Vmax,N_atoms)
        self.deltaz = (self.Vmax - self.Vmin) / (self.N_atoms - 1)
        super(categorical_DQN,self).__init__(
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

    def create_nets(self):
        self.state = tf.placeholder(tf.float32,[None,self.state_dim],name='state')
        self.action = tf.placeholder(tf.int32,[None],name='action')
        self.next_state = tf.placeholder(tf.float32,[None,self.state_dim],name='next_state')
        
        w_initializer,b_initializer = tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)
        # input:state
        # output:batch_size * action_dim * N_atoms p_values
        with tf.variable_scope('online_net'):
            e1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            e2 = tf.layers.dense(e1,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e2')
            e3 = tf.layers.dense(e2,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e3')
            # get Q(state)
            self.q_eval = tf.layers.dense(e3,self.action_dim * self.N_atoms,kernel_initializer=w_initializer,bias_initializer=b_initializer)
            p_eval = tf.reshape(self.q_eval,[-1,self.action_dim,self.N_atoms])
            self.p_eval = tf.nn.softmax(p_eval,axis=2)

        # get p for each action,p(state,action)
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0],dtype=tf.int32),self.action],axis=1)
        self.p_eval_w = tf.gather_nd(params=self.p_eval,indices=a_indices)

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t1')
            t2 = tf.layers.dense(t1,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t2')
            t3 = tf.layers.dense(t2,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='t3')
            # get Q(state)
            self.q_target = tf.layers.dense(t3,self.action_dim * self.N_atoms,kernel_initializer=w_initializer,bias_initializer=b_initializer)
            p_target = tf.reshape(self.q_target,[-1,self.action_dim,self.N_atoms])
            self.p_target = tf.nn.softmax(p_target,axis=2)


        self.p_target_w = tf.gather_nd(params=self.p_target,indices=a_indices)
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
        self.m = tf.placeholder(tf.float32,[None,self.N_atoms],name='m')
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(self.m,tf.math.log(self.p_eval_w)),axis=1))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
    
    def _training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)
        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]
        # get current p(s',a) 
        p_eval = self.sess.run(self.p_eval,feed_dict={self.state:next_state_batch})
        # find a* = argmax(p(s',a))
        a_ = []
        for i in range(self.batch_size):
            Q_ = []
            for j in range(self.action_dim):
                Q = np.sum(np.multiply(p_eval[i][j],self.z))
                Q_.append(Q)
            a_.append(np.argmax(Q_))

        Tz = np.zeros((self.batch_size,self.N_atoms))
        m = np.zeros((self.batch_size,self.N_atoms))
        # get p_(s',a*)
        p_ = self.sess.run(self.p_target_w,feed_dict={self.state:next_state_batch,self.action:a_})
        for i in range(self.batch_size):
            b = np.zeros((self.N_atoms))
            l = np.zeros((self.N_atoms)).astype(int)
            u = np.zeros((self.N_atoms)).astype(int)
            pp = p_[i]
            for j in range(self.N_atoms):
                # get Tz onto support z
                Tz[i][j] = reward_batch[i] + (1-done_batch[i]) * self.gamma * self.z[j]
                Tz[i][j] = np.clip(Tz[i][j],self.Vmin,self.Vmax)
                b[j] = (Tz[i][j] - self.Vmin) / self.deltaz
                l[j] = np.floor(b[j])
                u[j] = np.ceil(b[j])
                # get m vector
                m[i][l[j]] += pp[j] * (u[j] - b[j] + 0.) 
                m[i][u[j]] += pp[j] * (b[j] - l[j] + 0.)
        # train the net
        self.sess.run(self._train_op,feed_dict={self.state:state_batch,self.action:action_batch,self.m:m})

    def choose_action(self,state):
        state = state[np.newaxis,:]

        if np.random.uniform() > self.epsilon:
            p_eval = self.sess.run(self.p_eval,feed_dict={self.state:state})[0]
            Q_ = []
            for j in range(self.action_dim):
                Q = np.sum(np.multiply(p_eval[j],self.z))
                Q_.append(Q)
            return np.argmax(Q_)
        else:
            res = np.random.randint(0,self.action_dim)

        self.epsilon -= 0.001
        return res    

    def action_direct(self,state):
        state = state[np.newaxis,:]
        p_eval = self.sess.run(self.p_eval,feed_dict={self.state:state})[0]
        Q_ = []
        for j in range(self.action_dim):
            Q = np.sum(np.multiply(p_eval[j],self.z))
            Q_.append(Q)
        return np.argmax(Q_)      