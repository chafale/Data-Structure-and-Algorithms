from ast import List
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math


# 1. Array With Elements Not Equal to Average of Neighbours
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




# 2.  Rotate Array
"""
https://leetcode.com/problems/rotate-array/
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




# 3. Number of Subsequences That Satisfy the Given Sum Condition
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
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
            nums.sort()
            l, r = 0, len(nums) - 1
            res = 0
            mod = 10**9 + 7
            while l <= r:
                if nums[l] + nums[r] > target:
                    r -= 1
                else:
                    res += pow(2, r - l, mod)
                    l += 1
            return res % mod




# 4. Search in Rotated Sorted Array
"""
https://leetcode.com/problems/search-in-rotated-sorted-array/
Given the array nums after the possible rotation and an integer target, return the index of target 
if it is in nums, or -1 if it is not in nums.
You must write an algorithm with O(log n) runtime complexity.

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
"""
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1

        while l <= r:
            mid = (l + r)//2

            if target == nums[mid]:
                return mid
            
            # left sorted portion
            if nums[l] <= nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1

            # right sorted portion
            else:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1

        return -1
    



# 5. Search in Rotated Sorted Array II
"""
https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
There is an integer array nums sorted in non-decreasing order 
not necessarily with distinct values). [** has same values **]

`nums` is rotated at an unknown pivot index k.

Given the array nums after the rotation and an integer target, return true if target is in nums, 
or false if it is not in nums.

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
"""
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        l, r = 0, len(nums)-1

        while l <= r:
            mid = (l + r)//2

            if target == nums[mid]:
                return True

            # left sorted portion
            if nums[l] < nums[mid]:
                if nums[l] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1

            # right sorted portion
            elif nums[l] > nums[mid]:
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                l += 1
    
        return False




# 6. Find Minimum in Rotated Sorted Array
"""
https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/
Given the sorted rotated array nums of unique elements, return the minimum element of this array.
You must write an algorithm that runs in O(log n) time.

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
"""
class Solution:
    def findMin(self, nums: List[int]) -> int:
        start , end = 0 ,len(nums) - 1 
        curr_min = float("inf")
        
        while start  <  end :
            mid = (start + end ) // 2
            curr_min = min(curr_min, nums[mid])
            
            # right has the min 
            if nums[mid] > nums[end]:
                start = mid + 1
                
            # left has the  min 
            else:
                end = mid - 1 
                
        return min(curr_min,nums[start])




# 7. Single Element in a Sorted Array
"""
https://leetcode.com/problems/single-element-in-a-sorted-array
You are given a sorted array consisting of only integers where every element appears 
exactly twice, except for one element which appears exactly once.

Return the single element that appears only once.

Your solution must run in O(log n) time and O(1) space.

Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2

Input: nums = [3,3,7,7,10,11,11]
Output: 10
"""
# Approach 1 : O(n) TC using XOR
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        res = nums[0]
        for num in nums[1:]:
            res ^= num
        return res
    
# Approachh 2 : Binary Search O(log(n))
# https://www.youtube.com/watch?v=HGtqdzyUJ3k
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r)//2

            # check if mid is single element
            if (mid - 1 < 0 or nums[mid - 1] != nums[mid]) and \
               (mid + 1 == len(nums) or nums[mid + 1] != nums[mid]):
                return nums[mid]

            leftside = mid - 1 if nums[mid] == nums[mid - 1] else mid

            # check if left side elements are even ---
            # Note : single element will be present on the side where there are odd num of elements
            if leftside % 2 == 0:
                l = mid + 1
            else:
                r = mid - 1