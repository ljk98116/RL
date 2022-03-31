import math
import tensorflow as tf
import numpy as np
from collections import deque
from priority_replay_buffer import priority_replay_buffer
import random

class Rainbow_DQN:
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
        update_freq = 1000,
        # Distributional RL hyper parameters
        Vmin = -100,
        Vmax = 100,
        N_atoms = 31,

        # SumTree hyper parameters
        alpha = 0.6,
        e = 0.01,

        # n steps TD
        n_alpha = 0.01,
        n = 10
    ):
        self.params_cache = deque()
        self.env = env

        # replay buffer hyper parameters
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

        # support z gen
        self.N_atoms = N_atoms
        self.z = np.zeros((N_atoms))
        self.Vmax = Vmax
        self.Vmin = Vmin

        for i in range(self.N_atoms):
            self.z[i] = self.Vmin + i * (self.Vmax - self.Vmin) / (self.N_atoms - 1)
        
        # n steps TD
        self.n_alpha = n_alpha
        self.n = n

        # neoral net and training method
        self.create_nets()
        self.GetTz()
        self.training_method()
        self.sess.run(tf.global_variables_initializer())

    # Dueling_DQN net + softmax layer and update net op
    def create_nets(self):
        self.state = tf.placeholder(tf.float32,[None,self.state_dim],name='state')
        self.action = tf.placeholder(tf.int32,[None],name='action')
        self.next_state = tf.placeholder(tf.float32,[None,self.state_dim],name='next_state')
        
        w_initializer,b_initializer = tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)
        with tf.variable_scope('online_net'):
            e1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            # get Q(state)
            '''
            with tf.variable_scope('Value'):
                self.V = tf.layers.dense(e1,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Value')
            with tf.variable_scope('Advantage'):
                self.A = tf.layers.dense(e1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Advantage')
            
            self.q_eval = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims = True))
            # get Q for each action,Q(state,action)
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0],dtype=tf.int32),self.action],axis=1)
            self.q_eval_w = tf.gather_nd(params=self.q_eval,indices=a_indices)
            '''
            # get p_eval
            e2 = tf.layers.dense(e1,40,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Z_eval0')
            ze = tf.layers.dense(e2,self.action_dim * self.N_atoms,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Z_eval1')
            zer = tf.reshape(ze,[-1,self.action_dim,self.N_atoms])
            self.p_eval = tf.nn.softmax(zer,axis=2)

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.state,20,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='e1')
            # get Q(state)
            '''
            with tf.variable_scope('Value'):
                self.V2 = tf.layers.dense(t1,1,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Value')
            with tf.variable_scope('Advantage'):
                self.A2 = tf.layers.dense(t1,self.action_dim,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Advantage')
            
            self.q_target = self.V2 + (self.A2- tf.reduce_mean(self.A2,axis=1,keep_dims = True))
            '''
            # get p_target
            t2 = tf.layers.dense(t1,40,tf.nn.relu,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Z_target0')
            zt = tf.layers.dense(t2,self.action_dim * self.N_atoms,kernel_initializer=w_initializer,bias_initializer=b_initializer,name='Z_target1')
            ztr = tf.reshape(zt,[-1,self.action_dim,self.N_atoms])
            self.p_target = tf.nn.softmax(ztr,axis=2)

        # network copy ops
        online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='online_net')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='target_net')

        for var in online_params:
            self.holder_list.append(tf.placeholder(tf.float32,[1]+var.shape.as_list()))
            self.params_dict[self.holder_list[len(self.holder_list)-1]] = np.zeros(var.shape.as_list())
        
        for i,var in zip(range(len(target_params)),target_params):
            self.assign_ops.append(tf.assign(var,self.holder_list[i][0]))

        self.update_net_op = tf.group(*self.assign_ops)  
    
    def GetTz(self):
        with tf.variable_scope('GetTz'):
            self.Tz = tf.placeholder(tf.float32,[None,self.N_atoms],name='Tz')
            # compute Tz
            self.Tz2 = tf.clip_by_value(self.Tz,clip_value_min=self.Vmin+0.01,clip_value_max=self.Vmax-0.01)
            
    def training_method(self):
            self.m = tf.placeholder(tf.float32,[None,self.N_atoms],name='m')
            a_indices = tf.stack([tf.range(tf.shape(self.action)[0],dtype=tf.int32),self.action],axis=1)
            self.p_eval_w = tf.gather_nd(params=self.p_eval,indices=a_indices)
            self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(self.m,tf.math.log(self.p_eval_w)),axis=1))
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)   

    def choose_action(self,state):
        state = state[np.newaxis,:]

        if np.random.uniform() > self.epsilon:
            action = self.sess.run(self.p_eval,feed_dict={self.state:state})[0]
            res = np.argmax(np.sum(action,axis=1))
        else:
            res = np.random.randint(0,self.action_dim)

        self.epsilon -= 0.001
        return res

    def action_direct(self,state):
        state = state[np.newaxis,:]
        action = self.sess.run(self.p_eval,feed_dict={self.state:state})[0]
        res = np.argmax(np.sum(action,axis=1))
        return res
    
    def store_transition(self,state,action,reward,next_state,done):
        if len(self.replay_buffer) >= self.replay_size:
            self.replay_buffer.popleft()
        self.replay_buffer.append((state,action,reward,next_state,done))
    '''
    def store_transition(self,state,action,reward,next_state,done):
        state2 = state[np.newaxis,:]
        next_state2 = next_state[np.newaxis,:]
        action2 = action * np.ones((1))
        # eval and target values
        q_target = self.sess.run(self.q_target,feed_dict={self.state:next_state2})
        q_eval = self.sess.run(self.q_eval_w,feed_dict={self.state:state2,self.action:action2})
        # current TD error
        error = math.fabs(reward + self.gamma * np.argmax(q_target) - q_eval)
        self.replay_buffer.store_priority(error,state,action,reward,next_state,done)
    '''
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
        # print(self.time_step)
        if len(self.replay_buffer) > self.batch_size:
            # print("trained")
            self._training()
        if self.time_step % self.update_freq == 0:
            # print("updated")
            self.update_net()
    
    def _training(self):
        batch = random.sample(self.replay_buffer,self.batch_size)
        state_batch = [batch[i][0] for i in range(self.batch_size)]
        action_batch = [batch[i][1] for i in range(self.batch_size)]
        reward_batch = [batch[i][2] for i in range(self.batch_size)]
        next_state_batch = [batch[i][3] for i in range(self.batch_size)]
        done_batch = [batch[i][4] for i in range(self.batch_size)]
        # foreach member of batch,get a*
        a_ = []
        for i in range(self.batch_size):
            next_state = next_state_batch[i]
            next_state = next_state[np.newaxis,:]
            Q_a = []
            # 1 * action_dim * n_atoms
            p_target = self.sess.run(self.p_target,feed_dict={self.state:next_state})[0]
            # print(p_target[0][1])
            for j in range(self.action_dim):
                Q_a.append(np.sum(np.multiply(self.z,p_target[j])))
            # print(Q_a,np.argmax(Q_a))
            a_.append(np.argmax(Q_a))

        # get a*
        # print(a_)
        # get Tz
        # print(self.z)
        Tz = np.zeros((self.batch_size,self.N_atoms))
        for i in range(self.batch_size):
            for j in range(self.N_atoms):
                Tz[i][j] = reward_batch[i] - (done_batch[i] - 1) * self.gamma * self.z[j]
        
        Tz = self.sess.run(self.Tz2,feed_dict={self.Tz:Tz})
        # print(Tz)
        # get L,U,B
        b = np.zeros((self.batch_size,self.N_atoms))
        l = np.zeros((self.batch_size,self.N_atoms))
        u = np.zeros((self.batch_size,self.N_atoms))
        # print(Tz[0][0])
        for j in range(self.batch_size):
            for i in range(self.N_atoms):
                b[j][i] = (Tz[j][i] - self.Vmin + 0.) / ((self.Vmax - self.Vmin + 0.) / (self.N_atoms - 1 + 0.))
                l[j][i] = math.floor(b[j][i])
                u[j][i] = math.ceil(b[j][i])
        l,u = l.astype(np.int32),u.astype(np.int32)
        # print(b,l,u)

        # batch_size * N_atoms
        m = np.zeros((self.batch_size,self.N_atoms))
        for i in range(self.batch_size):
            tmpl = l[i]
            tmpu = u[i]
            tmpb = b[i]
            # get p(s_t+1,a*)
            next_state = next_state_batch[i]
            next_state = next_state[np.newaxis,:]
            # 1 * action_dim * n_atoms
            tmpp_ = self.sess.run(self.p_target,feed_dict={self.state:next_state})[0][a_[i]]
            # print(tmpp_)
            # print(tmpb)
            for j in range(self.N_atoms):
                # print(tmpp_[j],tmpl[j],tmpu[j])
                m[i][tmpl[j]] += tmpp_[j] * (tmpu[j]-tmpb[j] + 0.)
                m[i][tmpu[j]] += tmpp_[j] * (tmpb[j]-tmpl[j] + 0.)
        # print(m[0])
        
        # train_op
        _,loss,p_eval = self.sess.run([self._train_op,self.loss,self.p_eval],feed_dict={self.state:state_batch,self.m:m,self.action:action_batch})
        # print(loss)

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
        # print(tf.app.flags.FLAGS)
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
            # print(self.replay_buffer.p_buffer.total_times)
            if ep % 10 == 0:
                if self.evaluate(ep):
                    print('You Win !!')
                    break
    