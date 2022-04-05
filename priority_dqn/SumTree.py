from collections import deque
import numpy as np

class SumTree:
    def __init__(self,size):
        self.leaf = np.zeros((size))
        self.leaf_cursor = 0
        self.total_times = 0
        self.cursor = self.leaf_cursor + size - 1
        self.tree = np.zeros((2*size-1))
    
    def add(self,p):
        if self.leaf_cursor >= len(self.leaf):
            self.leaf_cursor = self.leaf_cursor % len(self.leaf)
        
        self.cursor = self.leaf_cursor + len(self.leaf) - 1
        change = p - self.tree[self.cursor]
        self.leaf[self.leaf_cursor] = p
        self.tree[self.cursor] = p

        self.update(change)
        self.leaf_cursor += 1
        self.total_times += 1

    def update(self,change):
        parent = self.cursor // 2
        while parent != 0:
            self.tree[parent] += change
            parent = parent // 2
        self.tree[parent] += change
        
    def traverse(self,value,id):
        if id >= len(self.leaf):
            return self.tree[id]
        if self.tree[2 * id + 1] > value:
            return self.traverse(value,2*id + 1)
        return self.traverse(value - self.tree[2*id + 1],2*id + 2)
