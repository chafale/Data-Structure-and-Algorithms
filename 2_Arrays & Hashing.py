from ast import List
import collections
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math
from functools import cmp_to_key


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
* * Good question
https://leetcode.com/problems/first-missing-positive/
Given an unsorted integer array nums, return the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.

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
#   3. then we will iterate through range [1 ... len(A)] and check if the element is present
#      -- to check if element is present go to index and check if integer present is negative
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
        
        for i in range(1, len(A) + 1):
            if A[i - 1] >= 0:
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




# 6. Good SubArray
"""
* * Good Problem
https://leetcode.com/problems/continuous-subarray-sum/
Given an integer array nums and an integer k, return true if nums has a 
good subarray or false otherwise.
A good subarray is a subarray where:
* its length is at least two, and
* the sum of the elements of the subarray is a multiple of k.

Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
"""
# Hard sol to come up but easy to understand
# we will use a hashmap with key: remainder and value : index 
# we will calculate prefix_sum 
# and to the prefix we mod by k -> we get remainder
# if the remainder is not in hashmap we add to hashmap with the index in hashmap
# if remainder is present in the hashmap ==> then we have obtained the subarray 
#       this is bcoz we can obtain same remainder iff we add multiple of k to the original
#       number whose remainder has collision
# now check the length of subarray is atleast two by comparing the two indexes 
# 
# one more thing we will initially add pair <0, -1> to hashmap to handle the edge cases 


# We are basically storing sum%k and storing it in the hashmap and checking it.
# Math logic is that the overall sum will get cancelled out because of modulo

class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        hashmap = {0: -1} # <remainder, index>

        prefix_sum = 0

        for i, n in enumerate(nums):
            prefix_sum += n
            remainder = prefix_sum % k
            if remainder not in hashmap:
                hashmap[remainder] = i
            elif i - hashmap[remainder] > 1:
                return True
            
        return False
            



# 7. Largest Number
"""
* * Good
https://leetcode.com/problems/largest-number/
Given a list of non-negative integers nums, arrange them such that they form the 
largest number and return it.

Input: nums = [10,2]
Output: "210"

Input: nums = [3,30,34,5,9]
Output: "9534330"
"""
# sol : greedy : sort according as if largest digit go first
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        def compare(n1, n2):
            # if we want n1 to go first we return -1
            # if we want n2 to go first we return 1
            # if equal return 0
            if n1 + n2 > n2 + n1:
                return -1
            else:
                return 1

        nums = [str(nums[i]) for i in range(len(nums))]
        nums = sorted(nums, key=cmp_to_key(compare))

        return str(int("".join(nums)))
    



# 8. Wiggle Sort
"""
* * Good
https://leetcode.com/problems/wiggle-sort/
Given an integer array nums, reorder it such that nums[0] <= nums[1] >= nums[2] <= nums[3]....

Input: nums = [3,5,2,1,6,4]
Output: [3,5,1,6,2,4]
Explanation: [1,6,2,5,3,4] is also accepted.

Input: nums = [6,6,5,6,3,8]
Output: [6,6,5,6,3,8]
"""
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        for i in range(1, len(nums)):
            # odd index
            if i % 2 == 1 and nums[i] < nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]
            # even index
            if i % 2 == 0 and nums[i] > nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]




# 9. Brick Wall
"""
https://leetcode.com/problems/brick-wall/
return the minimum number of crossed bricks after drawing such a vertical line.

Input: wall = [[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]]
Output: 2
"""
# Solution : create a hashmap for each-row find the gap 
# find the col with maximum gap and 
# return rows - max_gap --> which will return min bricks to cut
class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        gap_hashmap = {0:0}

        for row in wall:
            total = 0
            for brick in row[:-1]: # not inclusing the last brick
                total += brick
                gap_hashmap[total] = 1 + gap_hashmap.get(total, 0)

        return len(wall) - max(gap_hashmap.values())
    



# 10. Sort Colors - Dutch National Flag Problem
"""
* * Good
https://leetcode.com/problems/sort-colors/
Given an array nums with n objects colored red, white, or blue, sort them in-place so 
that objects of the same color are adjacent, with the colors in the order red, white, and blue.

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




# 11. Longest Consecutive Sequence
"""
* * Good problem
https://leetcode.com/problems/longest-consecutive-sequence/
Given an unsorted array of integers nums, return the length of the 
longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. 
Therefore its length is 4.

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
"""
# Solution : https://www.youtube.com/watch?v=P6RZZMu_maU
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # add nums to set
        numSet = set(nums)

        longest = 0
        for num in nums:
            # check if it is the start of the sequence
            if (num - 1) not in numSet:
                # calculate length of the sequence
                length = 0
                while (num + length) in numSet:
                    length += 1
                longest = max(longest, length)

        return longest
    



# 12. Product of Array Except Self
"""
* * Good Question
https://leetcode.com/problems/product-of-array-except-self/
Given an integer array nums, return an array answer such that answer[i] is equal to the 
product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
"""
# we will make use of prefix and postfix product concept
#           a  b  c  d  
#            \  \ i  /    
#             \  \  /
#  o/p:    prefix   postfix 
#                 []      
# 
# so prefix of (i-1 . . . 0) will be at i 
# and postfix of (i+1 . . . n) will be at i
# 
# we will have two pass : one for prefix and the other for postfix 
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * len(nums)

        # pass 1 : prefix
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix = prefix * nums[i]

        # pass 2 : postfix
        postfix = 1
        for i in range(len(nums)-1, -1, -1):
            res[i] = res[i] * postfix
            postfix = postfix * nums[i]

        return res




# 13. Top K Frequent Elements
"""
* * Good
https://leetcode.com/problems/top-k-frequent-elements/
Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
"""
# Approach 1 : count the occurrence of the interger in an array in hashmap 
# then use heap to find k most frequent element TC : k.log(N)
# 
# Approach 2 : Bucket sort 
# haphmap of key : feequency_count & value : list of nums having that frequency 
# The max limit of frequency is bounded by len(Array)
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums)
        freq_bucket = [[] for _ in range(len(nums) + 1)]

        for n, c in count.items():
            freq_bucket[c].append(n)

        res = []
        for i in range(len(freq_bucket)-1, -1, -1):
            for n in freq_bucket[i]:
                res.append(n)
                if len(res) == k:
                    return res
                



# 14. Maximum Number of Balloons
"""
* * Good
https://leetcode.com/problems/maximum-number-of-balloons/
Given a string text, you want to use the characters of text to form as many instances of 
the word "balloon" as possible.

You can use each character in text at most once. Return the maximum number of instances 
that can be formed.

Input: text = "nlaebolko"
Output: 1

Input: text = "loonbalxballpoon"
Output: 2
"""
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        countText = collections.Counter(text)
        balloon = collections.Counter("balloon")

        res = len(text)  # or float("inf")
        for c in balloon:
            res = min(res, countText[c] // balloon[c])
        return res
    
# (OR) the bottle-neck approach
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        balloonMap = {char: 0 for char in "balloon"}
        for char in text:
            if char in balloonMap:
                balloonMap[char] += 1
        
        balloonMap["o"] = balloonMap["o"] // 2
        balloonMap["l"] = balloonMap["l"] // 2

        bottleneck = min(balloonMap.values())

        return bottleneck



# 15. Find Pivot Index
"""
* * Good
https://leetcode.com/problems/find-pivot-index/description/
Given an array of integers nums, calculate the pivot index of this array.
The pivot index is the index where the sum of all the numbers strictly to the 
left of the index is equal to the sum of all the numbers strictly to the index's right.

Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The pivot index is 3.
Left sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11
Right sum = nums[4] + nums[5] = 5 + 6 = 11
"""
class Solution(object):
    """
    Let's say we knew S as the sum of the numbers, and we are at index i. 
    If we knew the sum of numbers leftsum that are to the left of index i, 
    then the other sum to the right of the index would just be S - nums[i] - leftsum.
    """
    def pivotIndex(self, nums):
        S = sum(nums)
        leftsum = 0
        for i, x in enumerate(nums):
            if leftsum == (S - leftsum - x):
                return i
            leftsum += x
        return -1
    



# 16. Majority Element - Boyer Moore Algorithm
"""
* * Good
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
    



# 17. Subarray Sum Equals K
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
    



# 18. Pairs of Songs With Total Durations Divisible by 60
"""
* Rare problem -- can skip in revision
You are given a list of songs where the ith song has a duration of time[i] seconds.

Return the number of pairs of songs for which their total duration in seconds is divisible by 60. 
Formally, we want the number of indices i, j such that i < j with (time[i] + time[j]) % 60 == 0.

Input: time = [30,20,150,100,40]
Output: 3
Explanation: Three pairs have a total duration divisible by 60:
(time[0] = 30, time[2] = 150): total duration 180
(time[1] = 20, time[3] = 100): total duration 120
(time[1] = 20, time[4] = 40): total duration 60

Input: time = [60,60,60]
Output: 3
Explanation: All three pairs have a total duration of 120, which is divisible by 60.
"""
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        hashmap = collections.defaultdict(int)
        res = 0
        for t in time:
            if t % 60 == 0:
                res += hashmap[0]
            else:
                res += hashmap[60 - (t % 60)]
            hashmap[t % 60] += 1
        return res




# 19. Minimum Time to Make Rope Colorful
"""
https://leetcode.com/problems/minimum-time-to-make-rope-colorful
Alice has n balloons arranged on a rope. You are given a 0-indexed string colors where 
colors[i] is the color of the ith balloon.

Alice wants the rope to be colorful. She does not want two consecutive balloons to be of the same 
color, so she asks Bob for help. Bob can remove some balloons from the rope to make it colorful. 
You are given a 0-indexed integer array neededTime where neededTime[i] is the time (in seconds) 
that Bob needs to remove the ith balloon from the rope.

Return the minimum time Bob needs to make the rope colorful.

Input: colors = "abaac", neededTime = [1,2,3,4,5]
Output: 3
Explanation: In the above image, 'a' is blue, 'b' is red, and 'c' is green.
Bob can remove the blue balloon at index 2. This takes 3 seconds.
There are no longer two consecutive balloons of the same color. Total time = 3.

Input: colors = "aabaa", neededTime = [1,2,3,4,1]
Output: 2
Explanation: Bob will remove the ballons at indices 0 and 4. Each ballon takes 1 second to remove.
There are no longer two consecutive balloons of the same color. Total time = 1 + 1 = 2.
"""

class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        timeCost = 0
        prevColor, prevTime = "#", 0 # <char, time>
        for color, time in zip(colors, neededTime):
            if color != prevColor:
                prevColor, prevTime = color, time
            else:
                timeCost += min(prevTime, time)
                prevTime = max(prevTime, time)
        return timeCost
    



# 20. Minimum Changes To Make Alternating Binary String
"""
https://leetcode.com/problems/minimum-changes-to-make-alternating-binary-string
You are given a string s consisting only of the characters '0' and '1'. In one 
operation, you can change any '0' to '1' or vice versa.

Return the minimum number of operations needed to make s alternating.

Input: s = "0100"
Output: 1
Explanation: If you change the last character to '1', s will be "0101", which is alternating.

Input: s = "1111"
Output: 2
Explanation: You need two operations to reach "0101" or "1010".
"""
class Solution:
    def minOperations(self, s: str) -> int:
        n = len(s)
        res = 0
        for i, char in enumerate(s):
            if int(char) != i % 2:
                res += 1
        return min(res, n - res)
    



# 21. Maximum Product Subarray
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
            curr = currMax * n
            currMax = max(curr, n * currMin, n)
            currMin = min(curr, n * currMin, n)
            res = max(res, currMax)

        return res
    



# 23. Merge Sorted Array
"""
* * Good
https://leetcode.com/problems/merge-sorted-array
You are given two integer arrays nums1 and nums2, sorted in non-decreasing order.
Merge nums1 and nums2 into a single array sorted in non-decreasing order.
m and n, representing the number of elements in nums1 and nums2.

Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]

Input: nums1 = [0], m = 0, nums2 = [1], n = 1
Output: [1]
"""
# Hint : start merging nums2 in nums1 in reverse order i.e 
# start from last index in nums1 and find who is largest number to fit the last index
# -- use 3 pointers --
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2, last = m-1, n-1, len(nums1) - 1

        # merge in reverse order
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[last] = nums1[p1]
                p1 -= 1
            else:
                nums1[last] = nums2[p2]
                p2 -= 1
            last -= 1

        # p1 is completed however p2 is still remaining
        while p2 >= 0:
            nums1[last] = nums2[p2]
            p2 -= 1
            last -=1  




# 24. Range Sum Query 2D - Immutable
"""
* * Good Question
https://leetcode.com/problems/range-sum-query-2d-immutable/
Given a 2D matrix matrix, handle multiple queries of the following type:
Calculate the sum of the elements of matrix inside the rectangle defined by its 
upper left corner (row1, col1) and lower right corner (row2, col2).

Implement the NumMatrix class:
1. NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
2. int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements 
   of matrix inside the rectangle defined by its upper left corner (row1, col1) and lower 
   right corner (row2, col2).

You must design an algorithm where sumRegion works on O(1) time complexity.
"""
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        ROWS, COLS = len(matrix), len(matrix[0])
        self.prefixMatrix = [[0] * (COLS + 1) for _ in range(ROWS + 1)]

        for r in range(ROWS):
            prefix = 0
            for c in range(COLS):
                prefix += matrix[r][c]
                above = self.prefixMatrix[r][c+1]
                self.prefixMatrix[r + 1][c + 1] = prefix + above

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        row1, col1, row2, col2 = row1 + 1, col1 + 1, row2 + 1, col2 + 1

        bottomLeft = self.prefixMatrix[row2][col2]
        top = self.prefixMatrix[row1 - 1][col2]
        left = self.prefixMatrix[row2][col1 - 1]
        topLeft = self.prefixMatrix[row1 - 1][col1 - 1]

        return bottomLeft - top - left + topLeft




# 25. Minimum Penalty for a Shop
"""
https://leetcode.com/problems/minimum-penalty-for-a-shop/
You are given the customer visit log of a shop represented by a 0-indexed 
string customers consisting only of characters 'N' and 'Y':
    -- if the ith character is 'Y', it means that customers come at the ith hour
    -- whereas 'N' indicates that no customers come at the ith hour.

If the shop closes at the jth hour (0 <= j <= n), the penalty is calculated as follows:
1. For every hour when the shop is open and no customers come, the penalty increases by 1.
2. For every hour when the shop is closed and customers come, the penalty increases by 1.

Return the earliest hour at which the shop must be closed to incur a minimum penalty.

Input: customers = "YYNY"
Output: 2
Explanation: 
- Closing the shop at the 0th hour incurs in 1+1+0+1 = 3 penalty.
- Closing the shop at the 1st hour incurs in 0+1+0+1 = 2 penalty.
- Closing the shop at the 2nd hour incurs in 0+0+0+1 = 1 penalty.
- Closing the shop at the 3rd hour incurs in 0+0+1+1 = 2 penalty.
- Closing the shop at the 4th hour incurs in 0+0+1+0 = 1 penalty.
Closing the shop at 2nd or 4th hour gives a minimum penalty. 
Since 2 is earlier, the optimal closing time is 2.
"""
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        #     Intution for this problem :-
        #     consider the case :
        #         N Y N Y Y
        #             j

        #         if the shop is close at j-th hour then 
        #             it is also closed for j+1, j+2, . . . hours

        #         so for the penalty ==> count no. of N before j and 
        #                                 (add) count no. of Y after j

        #         Note : here answer can be index n + 1 as well <edge case>
        
        n = len(customers)

        prefix_n = [0] * (n + 1)
        postfix_y = [0] * (n + 1)

        # calculating prefix of "N"
        n_soFar = 0
        for i in range(n + 1):
            prefix_n[i] = n_soFar
            if i < n and customers[i] == "N":
                n_soFar += 1

        # calculating postfix of "Y"
        y_soFar = 0
        for i in range(n - 1, -1, -1):
            if customers[i] == "Y":
                y_soFar += 1
            postfix_y[i] = y_soFar

        # calculating penalty
        minIdx, minPenalty = n, n
        for i in range(n + 1):
            penalty = prefix_n[i] + postfix_y[i]
            if penalty < minPenalty:
                minPenalty = penalty
                minIdx = i

        return minIdx




# 26. Valid Sudoku
"""
https://leetcode.com/problems/valid-sudoku/description/
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
"""
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = collections.defaultdict(set)
        cols = collections.defaultdict(set)
        squares = collections.defaultdict(set)
        
        for r in range(9):
            for c in range(9):
                val = board[r][c]

                if val == ".":
                    continue

                if (val in rows[r] or
                    val in cols[c] or
                    val in squares[(r//3, c//3)]):
                    return False

                rows[r].add(val)
                cols[c].add(val)
                squares[(r//3, c//3)].add(val)

        return True
    



# 27. Rotate Image
"""
https://leetcode.com/problems/rotate-image/description/
You are given an n x n 2D matrix representing an image, 
rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify 
the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
"""
# Solution 1 : 
# 1. Do Transpose of the matrix (i.e change col into rows and rows into cols) 
# 2. reverse each row  
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        row = len(matrix)
        
        # Transpose the matrix
        for i in range(row):
            for j in range(i + 1):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        # Reverse each row
        for i in range(row):
            matrix[i].reverse()