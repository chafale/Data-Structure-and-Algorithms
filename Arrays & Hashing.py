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




# 3. Degree of an Array
"""
https://leetcode.com/problems/degree-of-an-array
The degree of this array is defined as the maximum frequency of any one of its elements.
Find the smallest possible length of a (contiguous) subarray of nums, that has the same 
degree as nums.

Example 1:
Input: nums = [1,2,2,3,1]
Output: 2
Explanation: 
The input array has a degree of 2 because both elements 1 and 2 appear twice.
Of the subarrays that have the same degree:
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
The shortest length is 2. So return 2.

Example 2:
Input: nums = [1,2,2,3,1,4,2]
Output: 6
Explanation: 
The degree is 3 because the element 2 is repeated 3 times.
So [2,2,3,1,4,2] is the shortest subarray, therefore returning 6.
"""
# Solution : 
# An array that has degree d, must have some element x occur d times. 
# If some subarray has the same degree, then some element x (that occured d times), 
# still occurs d times. 
# The shortest such subarray would be from the first occurrence of x until the last occurrence. 

# For each element in the given array, let's know "left", the index of its first occurrence; 
# and "right", the index of its last occurrence. 
# Then, for each element x that occurs the maximum number of times, 
# right[x] - left[x] + 1 will be our candidate answer, 
# and we'll take the minimum of those candidates.
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        left, right, count = {}, {}, {}
        for i, n in enumerate(nums):
            if n not in left:
                left[n] = i
            right[n] = i
            count[n] = count.get(n, 0) + 1

        ans = len(nums)
        degree = max(count.values())
        for k in count:
            if count[k] == degree:
                arr_len = right[k] - left[k] + 1
                ans = min(ans, arr_len)

        return ans