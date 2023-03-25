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




