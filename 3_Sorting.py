from ast import List
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math


# Problem 1 - Array With Elements Not Equal to Average of Neighbors
"""
https://leetcode.com/problems/array-with-elements-not-equal-to-average-of-neighbors/
You are given a 0-indexed array nums of distinct integers. 
You want to rearrange the elements in the array such that every element in the rearranged array 
is not equal to the average of its neighbors.

More formally, the rearranged array should have the property such that for every i in the 
range 1 <= i < nums.length - 1, (nums[i-1] + nums[i+1]) / 2 is not equal to nums[i].

Return any rearrangement of nums that meets the requirements.
Input: nums = [1,2,3,4,5]
Output: [1,2,4,5,3]
Explanation:
When i=1, nums[i] = 2, and the average of its neighbors is (1+4) / 2 = 2.5.
When i=2, nums[i] = 4, and the average of its neighbors is (2+5) / 2 = 3.5.
When i=3, nums[i] = 5, and the average of its neighbors is (4+3) / 2 = 3.5.

Input: nums = [6,2,0,9,7]
Output: [9,7,6,2,0]
Explanation:
When i=1, nums[i] = 7, and the average of its neighbors is (9+6) / 2 = 7.5.
When i=2, nums[i] = 6, and the average of its neighbors is (7+2) / 2 = 4.5.
When i=3, nums[i] = 2, and the average of its neighbors is (6+0) / 2 = 3.
"""
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        """
        sort the i/p array
        take the first half and insert at alternate (odd indices)
        take second half and starting from left insert at the remaining positions
        """
        
        nums.sort()
        res = []

        l ,r = 0, len(nums)-1
        while len(res) != len(nums):
            res.append(nums[l])
            l += 1

            if l < r:
                res.append(nums[r])
                r -= 1

        return res




# Problem 2 - Rotate Array
"""
Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
"""
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # since k can be greater than len(nums) therefore we mod with the len(nums)
        k = k % len(nums)

        # reverse the nums array
        l, r = 0, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1
        
        # reverse the first k elements
        l, r = 0, k - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1
            
        # reverse the elements after k
        l, r = k, len(nums) - 1
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l, r = l + 1, r - 1




# Problem 3 - Number of Subsequences That Satisfy the Given Sum Condition
"""
https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/
You are given an array of integers nums and an integer target.

Return the number of non-empty subsequences of nums such that the sum of the minimum and maximum 
element on it is less or equal to target. Since the answer may be too large, return it modulo 10^9 + 7.

Input: nums = [3,5,6,7], target = 9
Output: 4
Explanation: There are 4 subsequences that satisfy the condition.
[3] -> Min value + max value <= target (3 + 3 <= 9)
[3,5] -> (3 + 5 <= 9)
[3,5,6] -> (3 + 6 <= 9)
[3,6] -> (3 + 6 <= 9)
"""
def numSubseq(self, A, target):
        A.sort()
        l, r = 0, len(A) - 1
        res = 0
        mod = 10**9 + 7
        while l <= r:
            if A[l] + A[r] > target:
                r -= 1
            else:
                res += pow(2, r - l, mod)
                l += 1
        return res % mod




