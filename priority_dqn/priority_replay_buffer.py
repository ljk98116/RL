from SumTree import SumTree
import random
import numpy as np

class priority_replay_buffer:
    def __init__(self,size):
        # priority save
        # current count of the transitions
        self.cnt = 0
        # replay size
        self.size = size
        # sumtree to store priority values
        self.p_buffer = SumTree(size)
        
        # priority -> transition
        self.replay_buffer = {}

    def store_priority(self,state,action,reward,nextstate,done):
        if self.cnt < 1:
            p = 1
        else:
            p = self.p_buffer.maxp + 1
        self.cnt += 1
        self.cnt = self.cnt % self.size
        self.p_buffer.add(p,(state,action,reward,nextstate,done))

    def sample(self,batch_size):
        batch = []
        idx = []
        priority = []
        segment = self.p_buffer.tree[0] / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i+1)
            s = random.uniform(lo, hi)
            idx.append(self.p_buffer.traverse(s,0))
            retp = self.p_buffer.leaf[idx[-1]]             
            batch.append(retp[1:][0])
            priority.append(retp[0])
        return batch,idx,priority
    
    def modify(self,p,idx):
        self.p_buffer.modify(p,idx)
    
