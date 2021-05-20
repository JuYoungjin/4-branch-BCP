class Node:
    def __init__(self, left, right, fr, nbr):
        self.left = left
        self.right = right
        self.fr = fr
        self.nbr = nbr

class Tree:
    def __init__(self, fromCnt, ExCnt, proVar):
        self.fromCnt = fromCnt
        self.ExCnt = ExCnt
        self.proVar = proVar

    def create_node(self, fr, nbr):
        node = Node(left=None, right=None, fr=fr, nbr=nbr)
        return node

    def append(self, parent):
        left_num = parent.nbr // 2
        right_num = (parent.nbr + 1) // 2

        if left_num == 1 and right_num == 1:
            parent.left = self.create_node(fr=self.proVar[self.ExCnt], nbr=1)
            self.ExCnt += 1
            parent.right = self.create_node(fr=self.proVar[self.ExCnt], nbr=1)
            self.ExCnt += 1
        elif left_num == 1:
            parent.left = self.create_node(fr=self.proVar[self.ExCnt], nbr=1)
            self.ExCnt += 1
            parent.right = self.create_node(fr=self.fromCnt, nbr=right_num)
            self.fromCnt += right_num
            self.append(parent.right)
        else:
            parent.left = self.create_node(fr=self.fromCnt, nbr=left_num)
            self.fromCnt += left_num
            parent.right = self.create_node(fr=self.fromCnt, nbr=right_num)
            self.fromCnt += right_num
            self.append(parent.left)
            self.append(parent.right)

    def create_tree(self, fr, nbr):
        root = self.create_node(fr=fr, nbr=nbr)
        self.fromCnt = fr + nbr
        self.append(root)
        return root




















