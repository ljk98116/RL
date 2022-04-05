from SumTree import SumTree

if __name__ == "__main__":
    tree = SumTree(8)
    tree.add(3)
    tree.add(10)
    tree.add(12)
    tree.add(4)
    tree.add(1)
    tree.add(2)
    tree.add(8)
    tree.add(2)
    print(tree.traverse(24,0))
    k=0
    while(k < 8):
        tree.add(2)
        k+=1
    print(tree.traverse(24,0))