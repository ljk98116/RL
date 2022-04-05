from SumTree import SumTree
import random

class priority_replay_buffer:
    def __init__(self,size,a,e):
        self.p_buffer = SumTree(size)
        self.replay_buffer = {}
        self.a = a
        self.e = e

    def store_priority(self,error,state,action,reward,nextstate,done):
        self.p_buffer.add((error + self.e) ** self.a)
        self.replay_buffer[(error + self.e) ** self.a] = (state,action,reward,nextstate,done)

    def sample(self,batch_size):
        batch = []
        segment = self.p_buffer.tree[0] / batch_size
        print(segment)
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i+1)
            s = random.uniform(lo, hi)
            retp = 0
            if i==0:
                retp = self.p_buffer.traverse(s,0)
            else:
                retp = self.p_buffer.traverse(s,0)
            batch.append(self.replay_buffer[retp])
        # print(batch)
        return batch
    
    
