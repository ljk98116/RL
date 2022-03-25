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
        if len(self.p_buffer) < batch_size:
            return batch
        segment = self.p_buffer[0] / batch_size
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i+1)
            s = random.uniform(lo, hi)
            retp = self.p_buffer.traverse(s,0)
            batch.append(self.replay_buffer[retp])
        return batch
