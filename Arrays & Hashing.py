from ast import List
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math


# =======================================================================================================
# ===================================== ARRAYS & HASHING ================================================
# =======================================================================================================

# 1. Replace Elements with Greatest Element on Right Side
"""
https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/
Given an array arr, replace every element in that array with the greatest element 
among the elements to its right, and replace the last element with -1.
"""
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        rightMax = -1
        for i in range(len(arr) -1, -1, -1):
            newMax = max(rightMax, arr[i])
            arr[i] = rightMax
            rightMax = newMax
        return arr
    



# 2. Two Sum
"""
https://leetcode.com/problems/two-sum/
Given an array of integers nums and an integer target, 
return indices of the two numbers such that they add up to target.
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i