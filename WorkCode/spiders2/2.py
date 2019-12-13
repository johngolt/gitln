class Node(object):
    """ A Doubly-linked lists' node. """

    def __init__(self, data=None, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev


class DoublyLinkedList(object):
    def __init__(self):
        head = Node()
        tail = Node()
        self.head = head
        self.tail = tail
        self.head.next = tail
        self.tail.prev = head
        self.count = 0

    def append(self, data):
        """ Append an item to the list. """
        new_node = Node(data, None, None)
        prev = self.tail.prev
        prev.next, new_node.prev = new_node, prev
        self.tail.prev, new_node.next = new_node, self.tail
        self.count += 1

    def iter(self):
        """ Iterate through the list. """
        current = self.head.next  #note subtle change
        while current.next:
            val = current.data
            current = current.next
            yield val

    def reverse_iter(self):
        """ Iterate backwards through the list. """
        current = self.tail.prev
        while current.prev:
            val = current.data
            current = current.prev
            yield val

    def delete(self, data):
        """ Delete a node from the list. """
        current = self.head.next
        while current.next:
            if current.data == data:
                current.prev.next, current.next.prev = current.next, current.prev
                self.count -= 1
                return
            current = current.next

    def search(self, data):
        """Search through the list. Return True if data is found, otherwise False."""
        for node in self.iter():
            if data == node:
                return True
        return False

    def print_foward(self):
        """ Print nodes in list from first node inserted to the last . """
        for node in self.iter():
            print(node)

    def print_backward(self):
        """ Print nodes in list from latest to first node. """
        current = self.tail.prev
        while current.prev:
            print(current.data)
            current = current.prev

    def reverse(self):
        """ Reverse linked list. """
        prev, current = self.head, self.head.next
        while current:
            if prev is self.head:
                prev.next = None
            if current is self.tail:
                current.prev = None
            prev.prev, current.next, current, prev = current, prev, current.next, current
        self.head, self.tail = self.tail, self.head

    def __getitem__(self, index):
        if index > self.count - 1:
            raise Exception("Index out of range.")
        current = self.head.next  # Note subtle change
        for n in range(index):
            current = current.next
        return current.data

    def __setitem__(self, index, value):
        if index > self.count - 1:
            raise Exception("Index out of range.")
        current = self.head.next  # Note subtle change
        for n in range(index):
            current = current.next
        current.data = value


class ArithmetricProgression:
    def __init__(self, begin, step, end=None):
        self.begin = begin
        self.step = step
        self.end = end

    def __iter__(self):
        """ 强制转换成前面的加法算式得到的类型 """
        result = type(self.begin+self.step)(self.begin)
        forever = self.end is None
        index = 0
        while forever or result < self.end:
            yield result
            index += 1
            result = self.begin + self.step * index
import ctypes


class DynamicArray:
    """ A dynamic array class akin to a
    simplified python list """
    def __init__(self):
        """ Create an empty array """
        self._n = 0
        self._capacity = 1
        self._A = self._make_array(self._capacity)
  
    def __len__(self):
        # Return number of elements stored in the array
        return self._n
   
    def __getitem__(self, k):
        """ Return element at index k """
        if not 0 <= k < self._n:
            raise IndexError('invalid index')
        return self._A[k]

    def append(self, obj):
        """ Add object to end of the array """
        if self._n == self._capacity:
            self._resize(2*self._capacity)
        self._A[self._n] = obj
        self._n += 1
  
    def _resize(self, c):
        """ Resize internal array to capacity c. """
        B = self._make_array(c)
        for k in range(self._n):
            B[k] = self._A[k]
        self._A = B
        self._capacity = c
  
    def _make_array(self, c):
        """ Return new array with capacity c. """
        return (c * ctypes.py_object)()


class SequenceIterator:
    """ An iterator for any pyhon's sequence types """
    def __init__(self, sequence):
        """ Create an iterator for the given sequence """
        self._seq = sequence
        self._k = -1

    def __next__(self):
        """ Return the next element, or else raise
        StopeIteration error """
        self._k += 1
        if self._k < len(self._seq):
            return self._seq[self._k]
        else:
            raise StopIteration()
  
    def __iter__(self):
        """ By convention, an iterator must return iteself
        as an iterator """
        return self


class Node:
    """ A singly-linked node """
    def __init__(self, data=None):
        self.data = data
        self.next = None


class SinglyLinkedList:
    """ A singly-linked list. """
    def __init__(self):
        """ create an empty list """
        self.tail = None
        self.head = None
        self.count = 0

    def append(self, data):
        """ Append an item to the list """
        node = Node(data)
        if self.head:
            self.head.next = node
            self.head = node
        else:
            self.tail = node
            self.head = node
        self.count += 1

    def iter(self):
        """ Iterable through the list """
        current = self.tail
        while current:
            val = current.data
            current = current.next
            yield val

    def delete(self, data):
        """ Delete a node from the list """
        current = self.tail
        prev = self.tail
        while current:
            if current.data == data:
                if current = self.tail:
                    self.tail = current.next
                else:
                    prev.next = current.next
                self.count -= 1
                return
            prev = current
            current = current.next

    def search(self, data):
        """ search through the list.
        Return True if data was found else return False """
        for node in self.iter():
            if data == node:
                return True
        return False

    def __getitem__(self, index):
        if index > self.count - 1:
            raise IndexError('index out of range')
        current = self.tail
        for n in range(index):
            current = current.next
        return current.data

    def __setitem__(self, index, value):
        if index > self.count - 1:
            raise IndexError('index out of range')
        current = self.tail
        for n in range(index):
            current = current.next
        current.data = value
