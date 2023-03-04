from ast import List
import sys
import math
import random
from typing import Optional
import collections

# Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l ,r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else:
                return mid
        return -1




# 2. Random Pick with Weight
"""
https://leetcode.com/problems/random-pick-with-weight
You are given a 0-indexed array of positive integers w where w[i] 
describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index 
in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking 
an index i is w[i] / sum(w).

Input
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output
[null,1,1,1,1,0]
"""
class Solution:

    def __init__(self, w: List[int]):
        """
        :type w: List[int]
        """
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum

    def pickIndex(self) -> int:
        """
        :rtype: int
        """
        target = self.total_sum * random.random()
        # run a linear search to find the target zone
        for i, prefix_sum in enumerate(self.prefix_sums):
            if target < prefix_sum:
                return i
            



