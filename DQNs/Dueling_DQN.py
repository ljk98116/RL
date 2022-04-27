from DQNs.DQN import DQN
import random
import numpy as np
import tensorflow as tf

class Dueling_DQN(DQN):
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
        super(Dueling_DQN,self).__init__(
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
        with tf.variable_scope('online_net'):
            e1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            # get Q(state)
            with tf.variable_scope('Value'):
                self.V = tf.layers.dense(e1,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Value')
            with tf.variable_scope('Advantage'):
                self.A = tf.layers.dense(e1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Advantage')
            
            self.q_eval = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims = True))

        # get Q for each action,Q(state,action)
        a_indices = tf.stack([tf.range(tf.shape(self.action)[0],dtype=tf.int32),self.action],axis=1)
        self.q_eval_w = tf.gather_nd(params=self.q_eval,indices=a_indices)

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            # get Q(state)
            with tf.variable_scope('Value'):
                self.V2 = tf.layers.dense(t1,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Value')
            with tf.variable_scope('Advantage'):
                self.A2 = tf.layers.dense(t1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Advantage')
            
            self.q_target = self.V2 + (self.A2- tf.reduce_mean(self.A2,axis=1,keep_dims = True))

        # network copy ops
        online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='online_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')

        for var in online_params:
            self.holder_list.append(tf.placeholder(tf.float32,[1]+var.shape.as_list()))
            self.params_dict[self.holder_list[len(self.holder_list)-1]] = np.zeros(var.shape.as_list())
        
        for i,var in zip(range(len(target_params)),target_params):
            self.assign_ops.append(tf.assign(var,self.holder_list[i][0]))

        self.update_net_op = tf.group(*self.assign_ops)      