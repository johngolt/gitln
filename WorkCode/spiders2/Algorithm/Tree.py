from collections import deque
import math


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

    def binary(self, arr, term):  # 有序序列
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

    def linear_order(self, arr, term):
        for i, value in enumerate(arr):
            if value == term:
                return i
            if value > term:
                return None

    def linear_unorder(self, arr, term):
        for i, value in enumerate(arr):
            if value == term:
                return i
        return None


class SortSequence:

    def bubble(self, arr):
        for i in range(1, len(arr)):
            for j in range(0, len(arr)-i):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    def selection(self, arr):
        for i in range(len(arr) - 1):
            index = i
        for j in range(i+1, len(arr)):
            if arr[j+1] < arr[index]:
                index = j
        if i != index:
            arr[i], arr[index] = arr[index], arr[i]
        return arr
    
    def insertion(self, arr):
        for i in range(len(arr)):
            pre = i - 1
            current = arr[i]
            while pre >= 0 and arr[pre] > current:
                arr[pre + 1] = arr[pre]
                pre -= 1
            arr[pre + 1] = current
        return arr
    
    def shell(self, arr):
        import math
        gap = 1
        while (gap < len(arr)/3):
            gap = gap*3 + 1
        while gap > 0:
            for i in range(gap, len(arr)):
                temp = arr[i]
                j = i - gap
                while j >= 0 and arr[j] > temp:
                    arr[j+gap] = arr[j]
                    j -= gap
                arr[j + gap] = temp
            gap = math.floor(gap/3)
        return arr

    def merge(self, arr):
        if len(arr) < 2:
            return arr
        mid = math.floor(len(arr)/2)
        left, right = arr[:mid], arr[mid:]
        return self._merge(self.merge(left), self.merge(right))
    
    def _merge(self, left, right):
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        while left:
            result.append(left.pop(0))
        while right:
            result.append(right.pop(0))
        return result
    
    def quick(self, arr, left=None, right=None):
        left = 0 if not isinstance(left, int) else left
        right = len(arr) - 1 if not isinstance(right, int) else right
        if left < right:
            index = self.partition(arr, left, right)
            self.quick(arr, left, index-1)
            self.quick(arr, index+1, right)
        return arr
    
    def partition(self, arr, left, right):
        pivot = left
        index = pivot + 1
        i = index
        while i <= right:
            if arr[i] < arr[pivot]:
                arr[i], arr[index] = arr[index], arr[i]
                index += 1
            i += 1
        arr[pivot], arr[index-1] = arr[index-1], arr[pivot]
        return index - 1

    def counting(self, arr, maxvalue):
        bucketlen = maxvalue + 1
        bucket = [0]*bucketlen
        index = 0
        arrlen = len(arr)
        for i in range(arrlen):
            if not bucket[arr[i]]:
                bucket[arr[i]] = 0
            bucket[arr[i]] += 1
        for j in range(bucketlen):
            while bucket[j] > 0:
                arr[index] = j
                index += 1
                bucket[j] -= 1
        return arr

    def heap(self, arr):
        arrlen = len(arr)
        self.buildMaxHeap(arr)
        for i in range(len(arr)-1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            arrlen -= 1
            self.heapify(arr, 0, arrlen)
        return arr

    def buildMaxHeap(self, arr):
        arrlen = len(arr)
        for i in range(math.floor(len(arr)/2), -1, -1):
            self.heapify(arr, i, arrlen)

    def heapify(self, arr, i, arrlen):
        left = 2*i + 1
        right = 2*i + 2
        largest = i
        if left < arrlen and arr[left] > arr[largest]:
            largest = left
        if right < arrlen and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, largest, arrlen)