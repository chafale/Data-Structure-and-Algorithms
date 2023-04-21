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




# 3. Find the Index of the First Occurrence in a String
"""
https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string
Given two strings needle and haystack, return the index of the first occurrence of 
needle in haystack, or -1 if needle is not part of haystack.

Input: haystack = "sadbutsad", needle = "sad"
Output: 0
Explanation: "sad" occurs at index 0 and 6.
The first occurrence is at index 0, so we return 0.
"""
# solving using pointers
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        m, n = len(haystack), len(needle)

        for window_start in range(m - n + 1):
            for i in range(n):
                if needle[i] != haystack[window_start + i]:
                    break

                if i == m -1:
                    return window_start # first index
                
        return -1
    



# 4. Longest Common Prefix
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
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != strs[0][i]:
                    return res
            res += strs[0][i]
        return res




# 5. Check If a String Contains All Binary Codes of Size K
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
    



# 6. Minimum Number of Swaps to Make the String Balanced
"""
A string is called balanced if and only if:

It is the empty string, or
It can be written as AB, where both A and B are balanced strings, or
It can be written as [C], where C is a balanced string.
You may swap the brackets at any two indices any number of times.

Return the minimum number of swaps to make s balanced.

Input: s = "][]["
Output: 1
Explanation: You can make the string balanced by swapping index 0 with index 3.
The resulting string is "[[]]".

Input: s = "]]][[["
Output: 2
Explanation: You can do the following to make the string balanced:
- Swap index 0 with index 4. s = "[]][][".
- Swap index 1 with index 5. s = "[[][]]".
The resulting string is "[[][]]".
"""
# Here we will keep track of extra closing brackets; and we will do so by following steps
# we will add +1 if there is closing bracket and add -1 if there is closing bracket 
# we will also maintain max_closing bracket so far and 
# return the (max_closing + 1)/2 because every swap operation will reduce the unbalance count by 2 
# i.e each swap will get rid of two extra closing bracket 

class Solution:
    def minSwaps(self, s: str) -> int:
        closeCnt, maxCloseCnt = 0, 0

        for i in range(len(s)):
            if s[i] == "[":
                closeCnt -= 1
            if s[i] == "]":
                closeCnt += 1
            maxCloseCnt = max(closeCnt, maxCloseCnt)

        return (maxCloseCnt + 1)//2
    



# 7. Valid Parenthesis String
"""
* * Hard problem
https://leetcode.com/problems/valid-parenthesis-string
Given a string s containing only three types of characters: '(', ')' and '*', 
return true if s is valid.

'*' could be treated as a single right parenthesis ')' or a single left parenthesis 
'(' or an empty string "".

Input: s = "()"
Output: true

Input: s = "(*))"
Output: true


"""
# Greedy: O(n)
class Solution:
    def checkValidString(self, s: str) -> bool:
        leftMin, leftMax = 0, 0

        for c in s:
            if c == "(":
                leftMin, leftMax = leftMin + 1, leftMax + 1
            elif c == ")":
                leftMin, leftMax = leftMin - 1, leftMax - 1
            else:
                leftMin, leftMax = leftMin - 1, leftMax + 1
            if leftMax < 0:
                return False
            if leftMin < 0:  # required because -> s = ( * ) (
                leftMin = 0
        return leftMin == 0