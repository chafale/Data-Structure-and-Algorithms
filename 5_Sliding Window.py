from ast import List
from collections import defaultdict
from collections import deque 
from collections import Counter
import heapq
import math

# Problem 1 - Sliding Window Maximum
"""
* * HARD PROBLEM
You are given an array of integers nums, there is a sliding window of size k which is moving 
from the very left of the array to the very right. You can only see the k numbers in the window. 
Each time the sliding window moves right by one position.

Return the max sliding window.

Example 1:
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
"""
# Uses the concept of monotonic decreasing queue
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        result = []
        q = deque()  # q will store indices

        l, r = 0, 0
        while r < len(nums):
            # push the element to the queue
            # but before pushing check whether queue follows monotonic decreasing property
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            # now we push the element
            q.append(r)

            # if element is out-of bound in the sliding window
            # remove it from the queue
            if l > q[0]:
                q.popleft()

            # if window is of size k
            if (r + 1) >= k:
                result.append(nums[q[0]])
                l += 1

            r += 1

        return result




# 2. Minimum Window Substring
"""
https://leetcode.com/problems/minimum-window-substring
Given two strings s and t of lengths m and n respectively, return the minimum window 
substring of s such that every character in t (including duplicates) is included in the window. 
If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.
"""
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not t: return ""

        t_count = Counter(t)
        s_count = {k : 0 for k, _ in t_count.items()}

        having, needed = 0, len(t_count)

        res = ""
        minLength = math.inf

        l, r = 0, 0
        while r < len(s):
            if s[r] in t_count:
                s_count[s[r]] += 1
                # we will increment having iff both counter values are same
                if s_count[s[r]] == t_count[s[r]]:
                    having += 1

            while having == needed and l <= r:
                window_size = r - l + 1
                if window_size < minLength:
                    res = s[l:r+1]
                    minLength = window_size

                if s[l] in s_count:
                    s_count[s[l]] -= 1
                    if s_count[s[l]] < t_count[s[l]]:
                        having -= 1

                l += 1

            r += 1

        return res




# 3. Longest Repeating Character Replacement
"""
https://leetcode.com/problems/longest-repeating-character-replacement/
You are given a string s and an integer k. You can choose any character of the string 
and change it to any other uppercase English character. You can perform this operation 
at most k times.

Return the length of the longest substring containing the same letter you can get after 
performing the above operations.

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
"""
# Sliding window validation equation 
# windowLen - count(most_freq_char) <= k   
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0

        l = 0
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

            res = max(res, r - l + 1)
        return res




# 4. Longest Substring Without Repeating Characters
"""
https://leetcode.com/problems/longest-substring-without-repeating-characters/
Given a string s, find the length of the longest 
substring without repeating characters.

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
"""
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        charSet = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            res = max(res, r - l + 1)
        return res




# 5. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold
"""
https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/
Given an array of integers arr and two integers k and threshold, return the number of 
sub-arrays of size k and average greater than or equal to threshold.

Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
Output: 3
Explanation: Sub-arrays [2,5,5],[5,5,5] and [5,5,8] have averages 4, 5 and 6 respectively. 
All other sub-arrays of size 3 have averages less than 4 (the threshold).
"""
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        res = 0
        curSum = sum(arr[:k-1])

        for L in range(len(arr) - k + 1):
            curSum += arr[L + k - 1]
            if (curSum / k) >= threshold:
                res += 1
            curSum -= arr[L]
        return res




# 6. Contains Duplicate II
"""
https://leetcode.com/problems/contains-duplicate-ii/
Given an integer array nums and an integer k, return true if there are two 
distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

Input: nums = [1,2,3,1], k = 3
Output: true

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
"""
# Note here the sliding window size will be k + 1 bcoz of this condition abs(i - j) <= k
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window = set()
        L = 0

        for R in range(len(nums)):
            if R - L > k:
                window.remove(nums[L])
                L += 1
            if nums[R] in window:
                return True
            window.add(nums[R])
        return False




# 7. Best Time to Buy and Sell Stock
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a 
different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. 
If you cannot achieve any profit, return 0.

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
"""
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1 # l is buy & r is sell
        max_profit = 0

        while r < len(prices):
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                max_profit = max(profit, max_profit)
            else:
                l = r
            r += 1

        return max_profit 
