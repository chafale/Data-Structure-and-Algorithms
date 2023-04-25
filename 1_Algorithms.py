from ast import List
import sys
import math
from typing import Optional
import collections

# 1. Kadane's Algorithm
"""
https://leetcode.com/problems/maximum-subarray/
Given an integer array nums, find the subarray with the largest sum, and return its sum.

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
"""
def maximumSubarray(a):
    max_g = a[0]
    max_c = a[0]

    for i in range(1, len(a)):
        max_c = max(a[i], max_c + a[i])
        if max_c > max_g:
            max_g = max_c

    return max_g





# 2. Binary Search
"""
... ... ... ... ... ... ... ... ... ...
"""
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        l ,r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] > target:
                r = mid - 1
            elif nums[mid] < target:
                l = mid + 1
            else: # mid == target
                return True
        return False
"""
Time Complexity : O(log n)
"""




# 3. Merge Sort
"""
Divide and conquer technique

pseudo code:
    global array a[low : high]

    def MergeSort(low, high):
        if low < high:
            mid = (low + high)//2

            # divide step
            MergeSort(low, mid)
            MergeSort(mid+1, high)

            # conquer or combine step
            Merge(low, mid, high)

    def Merge(low, mid, high):
        # b -> temporary array just to store results
        # a -> is the global array a[low : mid] and a[mid + 1 : high]
        # k is pointer to b array
        # i is pointer to a[low : mid]
        # j is pointer to a[mid + 1 : high]

        k = low, i = low, j = mid

        while i <= mid and j <= high:
            if a[i] < a[j]:
                b[k] = a[i]
                i += 1
            else:
                b[k] = a[j]
                j += 1

            k += 1

        # add remaing items from a[low : mid] or a[mid + 1 : high] to b array

        if i > mid:
            while j <= high:
                b[k] = a[j]
                j += 1
                k += 1
        else:
            while i <= mid:
                b[k] = a[i]
                i += 1
                k += 1

        # copy b array to a array
        a[low : high] = b.copy()

"""
# Time complexity O(n.logn)



# 4. Dutch Nation Flag Problem
"""
https://leetcode.com/problems/sort-colors/
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
"""
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        pointer_0 = 0
        pointer_2 = len(nums)-1
        
        mid = 0
        while(mid <= pointer_2): 
            # mid = 2
            if nums[mid] == 2: 
                nums[mid], nums[pointer_2] = nums[pointer_2], nums[mid]
                pointer_2 -= 1
                # Note : we don't increment mid here

            # mid = 0
            elif nums[mid] == 0:
                nums[mid], nums[pointer_0] = nums[pointer_0], nums[mid]
                pointer_0 += 1
                mid += 1

            # mid = 1
            else:
                mid += 1




# 5. Majority Element - Boyer Moore Algorithm
"""
https://leetcode.com/problems/majority-element/
Given an array nums of size n, return the majority element.
Follow-up: Could you solve the problem in linear time and in O(1) space?

Input: nums = [2,2,1,1,1,2,2]
Output: 2

Input: nums = [3,2,3]
Output: 3
"""
class Solution:
    def majorityElement(self, nums):
        count = 0
        candidate = None

        for num in nums:
            # when count is zero change the candidate element
            if count == 0:
                candidate = num

            if num == candidate:
                count += 1
            else:
                count += -1

        return candidate