from collections import deque

class Node:
    def __init__(self, data):
        self.data = data
        self.right_child = None
        self.left_child = None

class TraversalTree:

    def breadth_first(self, root):
        queue = deque([root])
        self.result = []
        while len(queue) > 0:
            node = queue.popleft()
            self.result.append(node.data)
            if node.left_child:
                queue.append(node.left_child)
            if node.right_child:
                queue.append(node.right_child)
        return self.result

    def inorder(self, root):
        current = root
        self.result = []
        if current is None:
            return 
        self.inorder(current.left_child)
        self.result.append(current.data)
        self.inorder(current.right_child)
        return self.result

    def preorder(self, root):
        self.result = []
        current = root
        if current is None:
            return 
        self.result.append(current.data)
        self.preorder(current.left_child)
        self.preorder(current.right_child)
        return self.result
    
    def postorder(self, root):
        self.result = []
        current = root
        if current is None:
            return
        self.postorder(current.left_child)
        self.postorder(current.right_child)
        self.result.append(current.data)
        return self.result

class Tree:
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        node = Node(data)
        if self.root is None:
            self.root = node
        else:
            current = self.root
            parent = None
            while True:
                parent = current
                if node.data < parent.data:
                    current = current.left_child
                    if current is None:
                        parent.left_child = node
                else:
                    current = current.right_child
                    if current is None:
                        parent.right_child = node
    
    def search(self, data):
        current = self.root
        while True:
            if current is None:
                return None
            elif current.data == data:
                return data
            elif current.data > data:
                current = current.left_child
            else:
                current.right_child
    
    def get_node_with_parent(self, data):
        parent = None
        current = self.root
        if current is None:
            return (parent, None)
        while True:
            if current is None:
                break
            elif current.data == data:
                return (parent, current)
            elif current.data > data:
                parent = current
                current = current.left_child
            else:
                parent = current
                current = current.right_child
        return (parent, None)


class Search():
    def binary(arr, term):  # 有序序列
        size = len(arr) - 1
        first, last = 0, size
        while first <= last:
            mid = (first + last)//2
            if arr[mid] == term:
                return mid
            if term > arr[mid]:
                first = mid + 1
            else:
                last = mid - 1
    def linear_order(arr, term):
        for i, value in enumerate(arr):
            if value == term:
                return i
            if value > term:
                return None
    def linear_unorder(arr, term):
        for i, value in enumerate(arr):
            if value == term:
                return i
        return None

class SortSequence:
    def selection(self, arr):
        for i, value in enumerate(arr):
            for j in range(i+1, len(arr)):
                if arr[j] < arr[i]:
                    arr[i], arr[j] = arr[j], arr[i]
    
    def _partition(self, arr, first, last):
        pivot = arr[first]
        pass
    
    def quick(arr, first, last):
        pass
        
    