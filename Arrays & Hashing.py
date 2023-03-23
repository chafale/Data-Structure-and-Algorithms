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
Input: arr = [17,18,5,4,6,1]
Output: [18,6,6,6,1,-1]
Explanation: 
- index 0 --> the greatest element to the right of index 0 is index 1 (18).
- index 1 --> the greatest element to the right of index 1 is index 4 (6).
- index 2 --> the greatest element to the right of index 2 is index 4 (6).
- index 3 --> the greatest element to the right of index 3 is index 4 (6).
- index 4 --> the greatest element to the right of index 4 is index 5 (1).
- index 5 --> there are no elements to the right of index 5, so we put -1.
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
    



# 4. First Missing Positive
"""
https://leetcode.com/problems/first-missing-positive/
Given an unsorted integer array nums, return the smallest missing positive integer.

Input: nums = [1,2,0]
Output: 3

Input: nums = [3,4,-1,1]
Output: 2

Input: nums = [7,8,9,11,12]
Output: 1
"""
# Sol : 
# Approach 1 : 
#   The smallest missing positive will belong to range(1 . . . len(A)+1) 
#   so we will add elements in hashset and check if elements in range(1,len(A))
#   are present 
#   TC O(n); however SC : O(n) -> we are not happy with this
# 
# Approach 2 : Use the input array 
#   1. convert all negative integers in arr A to zero
#   2. for num in A:
#           i = abs(num) - 1
#           if 'i' is in bound of the array index i.e 0 <= i < len(A):
#               if A[i] is not negative:
#                   A[i] *= -1 # make it negative
#   3. then we will iterate through range [1 ... len(A)] and check it element is present
#      to check if element is present go to index and check if integer present is negative
#       for i in range(1, len(A)+1):
#           idx = i - 1
#           if A[idx] > 0:
#               that means i is present
#           else:
#               # this is the smallest missing number
#               return i    
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        A = nums
        for i in range(len(A)):
            if A[i] < 0:
                A[i] = 0
            
        for i in range(len(A)):
            val = abs(A[i])
            if 1 <= val <= len(A):
                if A[val - 1] > 0:
                    A[val - 1] *= -1
                elif A[val - 1] == 0:
                    A[val - 1] = -1 * (len(A) + 1)
        
        for i in range( 1, len(A)+ 1):
            if A[i -1] >= 0:
                return i
        
        return len(A) + 1
        
    def firstMissingPositive_2(self, nums: List[int]) -> int:
        new = set(nums)
        i = 1
        while i in new:
            i += 1
        return i




# 5. Non-decreasing Array
"""
https://leetcode.com/problems/non-decreasing-array/
Given an array nums with n integers, your task is to check if it could become 
non-decreasing by modifying at most one element.

Input: nums = [4,2,3]
Output: true
Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

Input: nums = [4,2,1]
Output: false
Explanation: You cannot get a non-decreasing array by modifying at most one element.
"""
# Sol : Greedy : We will decrese the left element and make it equal to right element 
# as much as possible  
# we will check [i] to [i + 1]
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        changed = False
        for i in range(len(nums)-1):
            if nums[i] <= nums[i+1]:
                continue
            
            if changed:
                return False
            
            # We want to decrese the left element 
            # [3, 4, 2]
            #     i  i+1
            if i==0 or nums[i+1] >= nums[i-1]:
                nums[i] = nums[i+1]
            else:
                nums[i+1] = nums[i]
            changed = True

        return True 




