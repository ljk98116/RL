from collections import deque
import numpy as np

class SumTree:
    def __init__(self,size):
        self.leaf = np.zeros((size),dtype=object)
        self.size = size
        self.leaf_cursor = 0
        self.total_times = 0
        self.cursor = self.leaf_cursor + size - 1
        self.tree = np.zeros((2*size-1))
        self.maxp = 0
    
    def add(self,p,transition):
        if self.leaf_cursor >= len(self.leaf):
            self.leaf_cursor = self.leaf_cursor % len(self.leaf)
        self.cursor = self.leaf_cursor + len(self.leaf) - 1
        change = p - self.tree[self.cursor]
        self.leaf[self.leaf_cursor] = [p,transition]
        self.tree[self.cursor] = p

        self.update(change)
        self.leaf_cursor += 1
        self.total_times += 1

        if self.total_times < self.size:
            priority = [self.leaf[i][0] for i in range(self.leaf_cursor)]
        else:
            priority = [self.leaf[i][0] for i in range(len(self.leaf))]
        
        self.maxp = max(priority)
        # print(self.leaf)

    def update(self,change):
        parent = self.cursor // 2
        while parent != 0:
            self.tree[parent] += change
            parent = parent // 2
        self.tree[parent] += change
        
    def modify(self,p,idx):
        change = p - self.leaf[idx][0]
        self.leaf[idx][0] = p
        self.tree[idx + len(self.leaf) - 1] = p

        priority = [self.leaf[i][0] for i in range(len(self.leaf))]
        self.maxp = max(priority)

        # print(self.leaf)
        self.update(change)
        
    def traverse(self,value,id):
        # print(id)
        if  (2 * id + 1) >= len(self.tree):
            return id - len(self.leaf) + 1
        if self.tree[2 * id + 1] >= value:
            return self.traverse(value,2*id + 1)
        return self.traverse(value - self.tree[2*id + 1],2*id + 2)
