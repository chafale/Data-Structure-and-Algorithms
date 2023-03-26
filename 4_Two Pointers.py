from ast import List
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math


# =======================================================================================================
# ===================================== TWO POINTERS ====================================================
# =======================================================================================================

# Problem 1 - Trapping Rain Water
"""
https://leetcode.com/problems/trapping-rain-water/
Given n non-negative integers representing an elevation map where the width of each bar is 1, 
compute how much water it can trap after raining.

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented 
by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 
6 units of rain water (blue section) are being trapped.
"""
# Approach 1 - using prefix and postfix max
class Solution:
    def trap(self, height: List[int]) -> int:
        left_max = [0] * len(height)
        right_max = [0] * len(height)

        max_so_far = 0
        for i, ht in enumerate(height):
            left_max[i] = max_so_far
            max_so_far = max(max_so_far, ht)

        max_so_far = 0
        for i in range(len(height)-1, -1, -1):
            ht = height[i]
            right_max[i] = max_so_far
            max_so_far = max(max_so_far, ht)

        water_trapped = 0
        for i in range(len(height)):
            water = min(left_max[i], right_max[i]) - height[i]
            if water > 0:
                water_trapped += water
        
        return water_trapped
    
# Approach 2 - using two pointers
class Solution:
    def trap(self, height: List[int]) -> int:
        
        l = 0
        r = len(height)-1
        
        leftmax = height[0]
        rightmax = height[-1]
        ans = 0
        
        while(l <= r):
            if height[l] <= height[r]:
                if height[l] < leftmax:
                    ans += leftmax - height[l]
                else:
                    leftmax = height[l]
                l += 1
                
            else:
                
                if height[r] < rightmax:
                    ans += rightmax - height[r] 
                else:
                    rightmax = height[r]
                
                r -= 1
                
        return ans




# Problem 2 - Boats to Save People
"""
https://leetcode.com/problems/boats-to-save-people
You are given an array people where people[i] is the weight of the ith person, 
and an infinite number of boats where each boat can carry a maximum weight of limit. 
Each boat carries at most two people at the same time, provided the sum of the weight 
of those people is at most limit.

Return the minimum number of boats to carry every given person.

Example 1:
Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)

Example 2:
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)

Example 3:
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)
"""
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        l ,r = 0, len(people) - 1
        no_of_boats = 0
        while l <= r:
            if people[l] + people[r] <= limit:
                no_of_boats += 1
                l += 1
                r -= 1
            else:
                no_of_boats += 1
                r -= 1
        return no_of_boats
    



# 3. Container With Most Water
"""
https://leetcode.com/problems/container-with-most-water/
You are given an integer array height of length n. There are n vertical lines drawn such that 
the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains 
the most water.

Return the maximum amount of water a container can store.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.
"""
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        res = 0

        while l < r:
            res = max(res, min(height[l], height[r]) * (r - l))
            if height[l] < height[r]:
                l += 1
            elif height[r] <= height[l]:
                r -= 1
        return res




# 4. Two Sum
"""
https://leetcode.com/problems/two-sum
Given an array of integers nums and an integer target, return indices of the 
two numbers such that they add up to target.
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
# Solution : 
# It turns out we can do it in one-pass. While we are iterating and inserting elements into 
# the hash table, we also look back to check if current element's complement already exists 
# in the hash table. If it exists, we have found a solution and return the indices immediately. 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i




# 5. Two Sum II - Input Array Is Sorted
"""
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, 
find two numbers such that they add up to a specific target number. 
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
"""
# Solution : 
# We use two indices, initially pointing to the first and the last element, respectively. 
# Compare the sum of these two elements with target. If the sum is equal to target, 
# we found the exactly only solution. If it is less than target, we increase the smaller 
# index by one. If it is greater than target, we decrease the larger index by one. 
# Move the indices and repeat the comparison until the solution is found. 
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            _sum = numbers[l] + numbers[r]

            if _sum == target:
                return [l+1, r+1]
            elif _sum < target:
                l += 1
            else:
                r -= 1

        return [-1, -1]
    



# 6. 3 Sum
"""
https://leetcode.com/problems/3sum
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] 
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
"""
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            # Skip positive integers
            if a > 0:
                break

            if i > 0 and a == nums[i - 1]:
                continue

            l, r = i + 1, len(nums) - 1
            while l < r:
                threeSum = a + nums[l] + nums[r]
                if threeSum > 0:
                    r -= 1
                elif threeSum < 0:
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    # this is done to remove duplicates
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
        return res




# 7. 4 - Sum
"""
https://leetcode.com/problems/4sum/
Given an array nums of n integers, return an array of all the unique 
quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

1. 0 <= a, b, c, d < n
2. a, b, c, and d are distinct.
3. nums[a] + nums[b] + nums[c] + nums[d] == target

You may return the answer in any order.
"""
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res = []

        quad = []
        def kSum(k, start, target):
            if k != 2:
                for i in range(start, len(nums) - k + 1): # here start = idx
                    if i > start and nums[i] == nums[i-1]:
                        continue
                    quad.append(nums[i])
                    kSum(k-1, i + 1, target - nums[i])
                    quad.pop()
                return

            # two sum -ii base case
            l, r = start, len(nums) - 1
            while l < r:
                sum = nums[l] + nums[r]
                if sum < target:
                    l += 1
                elif sum > target:
                    r -= 1
                else:
                    res.append(quad + [nums[l], nums[r]])
                    l += 1
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1

        kSum(4, 0, target)
        return res
    



# 8. Remove Duplicates from Sorted Array
"""
https://leetcode.com/problems/remove-duplicates-from-sorted-array/
Given an integer array nums sorted in non-decreasing order, 
remove the duplicates in-place such that each unique element appears only once. 
The relative order of the elements should be kept the same.

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
"""
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l, r = 1, 1
        while r < len(nums):
            if nums[r] != nums[r-1]:
                nums[l] = nums[r]
                l += 1
            r += 1

        return l
    



# 9. Move Zeroes
"""
https://leetcode.com/problems/move-zeroes/
Given an integer array nums, move all 0's to the end of it while maintaining 
the relative order of the non-zero elements.

Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
"""
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        l = 0
        for r in range(len(nums)):
            if nums[r] != 0:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1




# 10. Valid Palindrome - II
"""
https://leetcode.com/problems/valid-palindrome-ii/
Given a string s, return true if the s can be palindrome after 
deleting at most one character from it.

Input: s = "abca"
Output: true
Explanation: You could delete the character 'c'.
"""
class Solution:
    def validPalindrome(self, s: str) -> bool:
        l ,r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                skipL, skipR = s[l+1 : r+1], s[l:r]
                return (skipL == skipL[::-1] or skipR == skipR[::-1])
            l += 1
            r -= 1
        
        return True
    



