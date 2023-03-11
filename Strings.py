from ast import List
from collections import defaultdict
from collections import Counter
from collections import deque 
from typing import Optional
import heapq
import math

# =======================================================================================================
# ========================================== STRINGS ====================================================
# =======================================================================================================

# 1. Valid Anagram
"""
https://leetcode.com/problems/valid-anagram/
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

Input: s = "anagram", t = "nagaram"
Output: true
"""
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
    



# 2. Is Subsequence
"""
https://leetcode.com/problems/is-subsequence/
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

Example 1:
Input: s = "abc", t = "ahbgdc"
Output: true

Example 2:
Input: s = "axc", t = "ahbgdc"
Output: false
"""
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)




# 3. Longest Common Prefix
"""
https://leetcode.com/problems/longest-common-prefix/
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".

Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
"""