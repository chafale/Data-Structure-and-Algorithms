from ast import List
import sys
import math
from typing import Optional
import collections

# Kadane's Algorithm
"""
https://leetcode.com/problems/maximum-subarray/
Given an integer array nums, find the subarray with the largest sum, and return its sum.
"""
def maximumSubarray(a):
    max_g = a[0]
    max_c = a[0]

    for i in range(1, len(a)):
        max_c = max(a[i], max_c + a[i])
        if max_c > max_g:
            max_g = max_c

    return max_g


# Binary Search
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
"""
