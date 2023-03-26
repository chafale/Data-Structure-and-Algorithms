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
    



# 2. Encode and Decode TinyURL
"""
https://leetcode.com/problems/encode-and-decode-tinyurl/
TinyURL is a URL shortening service where you enter a URL
nd it returns a short URL such as http://tinyurl.com/4e9iAk
Design a class to encode a URL and decode a tiny URL.

Implement the Solution class:
Solution() Initializes the object of the system.
String encode(String longUrl) Returns a tiny URL for the given longUrl.
String decode(String shortUrl) Returns the original long URL for the given shortUrl. 
It is guaranteed that the given shortUrl was encoded by the same object.
"""
class Codec:
    def __init__(self):
        self.encodeMap = {}
        self.decodeMap = {}
        self.base = "http://tinyurl.com/"

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        if longUrl not in self.encodeMap: 
            shortUrl = self.base + str(len(self.encodeMap) + 1)
            self.encodeMap[longUrl] = shortUrl
            self.decodeMap[shortUrl] = longUrl
        return self.encodeMap[longUrl]

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.decodeMap[shortUrl]




# 3. Encode and Decode Strings
"""
https://leetcode.com/problems/encode-and-decode-strings/
Design an algorithm to encode a list of strings to a string. The encoded string is 
then sent over the network and is decoded back to the original list of strings.
"""
class Solution:
    """
    @param: strs: a list of strings
    @return: encodes a list of strings to a single string.
    """

    def encode(self, strs):
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res

    """
    @param: s: A string
    @return: decodes a single string to a list of strings
    """

    def decode(self, s):
        res, i = [], 0

        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res




# 4. Range Sum Query - Immutable
"""
https://leetcode.com/problems/range-sum-query-immutable/
Implement the NumArray class:
1. NumArray(int[] nums) Initializes the object with the integer array nums.
2. int sumRange(int left, int right) Returns the sum of the elements of nums between 
   indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).
"""
# concept of prefix sum
class NumArray:

    def __init__(self, nums: List[int]):
        self.prefix = []
        cur = 0
        for n in nums:
            cur += n
            self.prefix.append(cur)
        
    def sumRange(self, left: int, right: int) -> int:
        r = self.prefix[right] 
        l = self.prefix[left - 1] if left > 0 else 0
        return r - l




# 5. Maximum Frequency Stack
"""
https://leetcode.com/problems/maximum-frequency-stack/
Design a stack-like data structure to push elements to the stack and pop the most frequent element 
from the stack.

Implement the FreqStack class:
1. FreqStack() constructs an empty frequency stack.
2. void push(int val) pushes an integer val onto the top of the stack.
3. int pop() removes and returns the most frequent element in the stack.
    If there is a tie for the most frequent element, the element closest to the stack's top is 
    removed and returned.
"""
# we will seaprate each number in to bucket of their count : 
# and bucket will contain list of elements
# we will pop element from the max count bucket
class FreqStack:

    def __init__(self):
        self.frequency_map = {}  # we will have frequency count of elements
        self.maxCount = 0
        self.freq_cnt_grp = defaultdict(list)        

    def push(self, val: int) -> None:
        val_freq = 1 + self.frequency_map.get(val, 0)
        self.frequency_map[val] = val_freq
        if val_freq > self.maxCount:
            self.maxCount = val_freq
        self.freq_cnt_grp[val_freq].append(val)
        
    def pop(self) -> int:
        pop_elem = self.freq_cnt_grp[self.maxCount].pop()
        self.frequency_map[pop_elem] -= 1
        if not self.freq_cnt_grp[self.maxCount]:
            self.maxCount -= 1
        return pop_elem
    



# 6. Min Stack
"""
https://leetcode.com/problems/min-stack/
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.
"""
# Hint : Consider each node in the stack having a minimum value. 
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]
    



# 7. Time Based Key-Value Store
"""
Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.

void set(String key, String value, int timestamp) Stores the key key with the value value 
    at the given time timestamp.

String get(String key, int timestamp) Returns a value such that set was called previously, 
    with timestamp_prev <= timestamp. If there are multiple such values, it returns the value 
    associated with the largest timestamp_prev. 
    
If there are no values, it returns "".

Input
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
Output
[null, null, "bar", "bar", null, "bar2", "bar2"]

Explanation
TimeMap timeMap = new TimeMap();
timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
timeMap.get("foo", 1);         // return "bar"
timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
timeMap.get("foo", 4);         // return "bar2"
timeMap.get("foo", 5);         // return "bar2"
"""
class TimeMap:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyStore = {}  # key : list of [val, timestamp]

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.keyStore:
            self.keyStore[key] = []
        self.keyStore[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        res, values = "", self.keyStore.get(key, [])
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l + r) // 2
            if values[m][1] <= timestamp:
                res = values[m][0]
                l = m + 1
            else:
                r = m - 1
        return res