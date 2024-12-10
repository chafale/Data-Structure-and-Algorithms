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
def maxSubArray(self, nums: List[int]) -> int:
    curr_sum = nums[0]
    max_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(curr_sum + num, num)
        max_sum = max(curr_sum, max_sum)
    return max_sum




# 2. Binary Search
"""
... ... ... ... ... ... ... ... ... ...
Given a a list of numbers in sorted manner. Efficiently find the numbers
in the given list of numbers.
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
# Solution 1 : sort and the majority element is the element that appears more than ⌊n / 2⌋ times.
# Soultion 2 : as below
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




# 6. Maximum Product Subarray
"""
* * Good
https://leetcode.com/problems/maximum-product-subarray/
Given an integer array nums, find a subarray that has the largest product, 
and return the product.

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = nums[0]
        currMax, currMin = 1, 1

        for n in nums:
            # storing max result in temp so that currMax does get override in currMin calculation
            temp = max(n * currMax, n * currMin, n)
            currMin = min(n * currMax, n * currMin, n)
            currMax = temp
            res = max(res, currMax)
        return res
    



# 7. Subarray Sum Equals K
"""
* * Good Question
https://leetcode.com/problems/subarray-sum-equals-k
Given an array of integers nums and an integer k, return the total number of subarrays 
whose sum equals to k.

A subarray is a contiguous non-empty sequence of elements within an array.

Input: nums = [1,1,1], k = 2
Output: 2

Input: nums = [1,2,3], k = 3
Output: 2

https://youtu.be/fFVZt-6sgyo?t=298
"""
# Prefix map : <prefix, count>
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefixMap = {0 : 1}
        currSum = 0
        res = 0
        for num in nums:
            currSum += num
            if currSum - k in prefixMap:
                res += prefixMap[currSum - k]
            prefixMap[currSum] = prefixMap.get(currSum, 0) + 1
        return res