from collections import deque

class Stack:
    def __init__(self):
        self._d = []
    def push(self, item):
        self._d.append(item)
    def pop(self):
        return self._d.pop()
    def isEmpty(self):
        return not self._d

class Queue:
    def __init__(self):
        self._d = deque()
    def push(self, item):
        self._d.append(item)
    def pop(self):
        return self._d.popleft()
    def isEmpty(self):
        return not self._d
