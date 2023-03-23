from ast import List
from collections import defaultdict
from collections import Counter
from collections import deque 
from typing import Optional
import heapq
import math
import random



# 1. Insert Delete GetRandom O(1)
"""
Implement the RandomizedSet class:

RandomizedSet() Initializes the RandomizedSet object.
1. bool insert(int val) Inserts an item val into the set if not present. 
    Returns true if the item was not present, false otherwise.
2. bool remove(int val) Removes an item val from the set if present. 
    Returns true if the item was present, false otherwise.
3. int getRandom() Returns a random element from the current set of elements 
    (it's guaranteed that at least one element exists when this method is called). 
    Each element must have the same probability of being returned.

You must implement the functions of the class such that each function works in 
average O(1) time complexity.
"""
# Sol : we will maintain an hashmap and array
class RandomizedSet:

    def __init__(self):
        self.hashmap = {}
        self.list = []
        

    def insert(self, val: int) -> bool:
        res = val not in self.hashmap

        if res:
            self.hashmap[val] = len(self.list)
            self.list.append(val)

        return res

    def remove(self, val: int) -> bool:
        # copy the last value from the array to the positiion where the element has 
        # to be removed and pop() from the end of the queue 
        # also update the last element index in hashmap 

        res = val in self.hashmap

        if res:
            last_elem = self.list[-1]
            remove_elem_index = self.hashmap[val]

            self.list[remove_elem_index] = last_elem
            self.hashmap[last_elem] = remove_elem_index

            del self.hashmap[val]
            self.list.pop()

        return res
        

    def getRandom(self) -> int:
        return random.choice(self.list)
    



