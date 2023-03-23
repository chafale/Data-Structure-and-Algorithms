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




# 4. Check If a String Contains All Binary Codes of Size K
"""
https://leetcode.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/
Given a binary string s and an integer k, return true if every binary code of length k 
is a substring of s. Otherwise, return false.

Input: s = "00110110", k = 2
Output: true
Explanation: The binary codes of length 2 are "00", "01", "10" and "11". 
They can be all found as substrings at indices 0, 1, 3 and 2 respectively.

Input: s = "0110", k = 1
Output: true
Explanation: The binary codes of length 1 are "0" and "1", it is clear that 
both exist as a substring. 

Input: s = "0110", k = 2
Output: false
Explanation: The binary code "00" is of length 2 and does not exist in the array.
"""
# Solution : 
# number of binary code of lenght k = 2^k
# so we fill find all sunique substring of size k and if the count is equal to 2^k 
# we will return True  
class Solution:
    def hasAllCodes(self, s: str, k: int) -> bool:
        hashset = set()
        for i in range(len(s) - k + 1):
            hashset.add(s[i: i+k])

        return len(hashset) == (2 ** k)