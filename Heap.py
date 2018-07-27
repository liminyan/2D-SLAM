import heapq
import random
 
class MyHeap(object):
    def __init__(self, initial=None, key=lambda x:x):
        self.key = key
        self._data = []
    def push(self, item):
       
        heapq.heappush(self._data, ((item.len), item))
       
    def pop(self):
        if(len(self._data)>1):
            return heapq.heappop(self._data)[1]
        else:
            return None


















                        