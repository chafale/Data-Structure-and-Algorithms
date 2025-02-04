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

        for i in range(m - n + 1):
            if haystack[i:i+n] == needle:
                return i
                
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
        common_prefix = strs[0]

        for i in range(len(common_prefix)):
            # check whether char at index `i` is common prefix in all the strings
            for s in strs:
                if i == len(s) or s[i] != common_prefix[i]:
                    return res
            # add common_prefix[i] to the result
            res += common_prefix[i]

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
    



# 8. Break a Palindrome
"""
https://leetcode.com/problems/break-a-palindrome/
Given a palindromic string of lowercase English letters palindrome, replace exactly one character 
with any lowercase English letter so that the resulting string is not a palindrome and that it is the 
lexicographically smallest one possible.

Input: palindrome = "abccba"
Output: "aaccba"
Explanation: There are many ways to make "abccba" not a palindrome, such as "zbccba", "aaccba", and "abacba".
Of all the ways, "aaccba" is the lexicographically smallest.

Input: palindrome = "a"
Output: ""
Explanation: There is no way to replace a single character to make "a" not a palindrome, so return an empty 
string.
"""
# Algorithm:
"""
Algorithm:
1. If the length of the string is 1, return an empty string since we cannot create 
    a non-palindromic string in this case.
2. Iterate over the string from left to the middle of the string: if the character is not `a`, 
    change it to `a` and return the string.
3. If we traversed over the whole left part of the string and still haven't got a non-palindromic 
    string, it means the string has only a's. Hence, change the last character to `b` and return 
    the obtained string.
"""
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        if len(palindrome) == 1:
            return ""

        res = [char for char in palindrome]
        # only traverse to middle of array
        for i in range(len(palindrome)//2):
            if res[i] != "a":
                res[i] = "a"
                return "".join(res)

        res[-1] = "b"
        return "".join(res)
    



# 9. Interleaving String
"""
https://leetcode.com/problems/interleaving-string
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
Explanation: One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
Output: false
Explanation: Notice how it is impossible to interleave s2 with any other string to obtain s3.

Input: s1 = "", s2 = "", s3 = ""
Output: true
"""
# Approach 1 - Top down DP approach
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False

        dp = {}
        # here i -> s1 index; j -> s2 index
        def dfs(i, j):
            if i == len(s1) and j == len(s2):
                return True

            if (i, j) in dp:
                return dp[(i, j)]

            ans = False

            # pick ith char from s1
            if i < len(s1) and s1[i] == s3[i+j]:
                ans = ans or dfs(i+1, j)

            # pick jth char from s2
            if j < len(s2) and s2[j] == s3[i+j]:
                ans = ans or dfs(i, j+1)

            dp[(i, j)] = ans

            return ans

        return dfs(0, 0)
    
# Approach 2 - Bottom up approach

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        
        dp = [[False for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        dp[len(s1)][len(s2)] = True

        for i in range(len(s1), -1, -1):
            for j in range(len(s2), -1, -1):
                ans = False
                
                # pick ith char from s1
                if i < len(s1) and s1[i] == s3[i+j] and dp[i+1][j]:
                    dp[i][j] = True

                # pick jth char from s2
                if j < len(s2) and s2[j] == s3[i+j] and dp[i][j+1]:
                    dp[i][j] = True

        return dp[0][0]
    



# 10. Longest Palindromic Substring
"""
Given a string s, return the longest palindromic substring in s.

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Input: s = "cbbd"
Output: "bb"
"""
# Approach : start from middle and expand  . . . 
# video : https://www.youtube.com/watch?v=XYQecbcd6_c
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ""
        resLen = 0

        for i in range(len(s)):
            # odd length palindrome
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    resLen = (r - l + 1)
                    res = s[l : r + 1]
                l -= 1
                r += 1

            # even length palindrome
            l, r = i, i+1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    resLen = (r - l + 1)
                    res = s[l : r + 1]
                l -= 1
                r += 1

        return res
    


# 11. Palindromic Substrings
"""
https://leetcode.com/problems/palindromic-substrings/
Given a string s, return the number of palindromic substrings in it.

Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
"""
class Solution:
    def countSubstrings(self, s: str) -> int:
        count = 0

        for i in range(len(s)):
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                count += 1
                l -= 1
                r += 1

            l, r = i, i+1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                count += 1
                l -= 1
                r += 1

        return count
    



# 12. Longest Substring with At Most Two Distinct Characters
"""
https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters
Given a string s, return the length of the longest 
substring that contains at most two distinct characters.
Input: s = "eceba"
Output: 3
Explanation: The substring is "ece" which its length is 3.

Input: s = "ccaabbb"
Output: 5
Explanation: The substring is "aabbb" which its length is 5.
"""
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        n = len(s)
        if n < 3:
            return n

        l, r = 0, 0
        hashMap = {}

        max_len = 2

        while r < len(s):
            hashMap[s[r]] = r

            if len(hashMap) == 3:
                del_idx = min(hashMap.values())
                del hashMap[s[del_idx]]
                l = del_idx + 1

            max_len = max((r - l + 1), max_len)
            r += 1
        
        return max_len
    



# 13. Group Anagrams
"""
https://leetcode.com/problems/group-anagrams/
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
"""
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        for word in strs:
            count = [0] * 26
            for char in word:
                count[ord(char) - ord('a')] += 1
            res[tuple(count)].append(word)
        return res.values()
    



# 14. Word Break
"""
https://leetcode.com/problems/word-break/
Given a string s and a dictionary of strings wordDict, return true if s can be segmented 
into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
"""
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True

        for i in range(len(s)-1, -1, -1):
            for w in wordDict:
                if (i + len(w)) <= len(s) and s[i:i+len(w)] == w:
                    dp[i] = dp[i + len(w)]
                if dp[i]:
                    break

        return dp[0]
    



# 15. Decode String
"""
https://leetcode.com/problems/decode-string/
Given an encoded string, return its decoded string.
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Input: s = "3[a2[c]]"
Output: "accaccacc"
"""
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []

        for char in s:
            if char is not "]":
                stack.append(char)
            else:
                sub_str = ""
                while stack[-1] is not "[":
                    sub_str = stack.pop() + sub_str
                stack.pop()

                multiplier = ""
                while stack and stack[-1].isdigit():
                    multiplier = stack.pop() + multiplier

                stack.append(int(multiplier) * sub_str)

        return "".join(stack)
    



# 16. Count Binary Substrings
"""
* * Ticky question
https://leetcode.com/problems/count-binary-substrings/
Given a binary string s, return the number of non-empty substrings that have 
the same number of 0's and 1's, and all the 0's and all the 1's in these substrings 
are grouped consecutively.

Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 
1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
"""
# Hint : To solve this problem just ignore the example and consider the scenario below: 
# 
# How many substrings are present in the below string with equal 0's and 1's?
#          0 0 0 1 1 1
#              ---
#            -------
#          -----------
# Answer = 3
# 
# Now if we add one more zero to it :
#         0 0 0 0 1 1 1
#               ---
#             -------
#           ----------- 
# 
# Answer = Min(countZeros, countOnes) = 3
# 
# Algorithm : here we will maintain two variables prevRunOfChar and currRunOfChar
# and we will be staring the loop from 1st index
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        res = 0

        prevRunOfChar = 0 
        currRunOfChar = 1 # bcoz we are staring from 1st index and to coust char at 0th index

        i = 1
        while i < len(s):
            # 1 1 0 0 0 1 1 0    OR     1 1 0 0 0 1 1 0       
            # -----                     -----------
            #     i                               i
            if s[i-1] != s[i]:
                res += min(prevRunOfChar, currRunOfChar)
                prevRunOfChar = currRunOfChar
                currRunOfChar = 1
            else:
                currRunOfChar += 1

            i += 1
        
        # finally at the end of string
        res += min(prevRunOfChar, currRunOfChar)
        return res
    



# 17. String Compression
"""
https://leetcode.com/problems/string-compression/
Given an array of characters chars, compress it using the following algorithm:
chars = ["a","a","b","b","c","c","c"]
Output : 6 because the len of arr ["a","2","b","2","c","3"] is 6

Do the operation in-place.

and Input: chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
Output: Return 4, and the first 4 characters of the input array should be: ["a","b","1","2"].
"""
class Solution:
    def compress(self, chars: List[str]) -> int:
        index = 0
        
        i = 0
        while i < len(chars):
            j = i
            while j < len(chars) and chars[i] == chars[j]:
                j += 1

            chars[index] = chars[i]
            index += 1

            if (j - i) > 1:
                numStr = str(j - i)
                for c in numStr:
                    chars[index] = c
                    index += 1
    
            i = j

        return index
    



# 18. Minimum Remove to Make Valid Parentheses
"""
https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses
Given a string s of '(' , ')' and lowercase English characters.
Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) 
so that the resulting parentheses string is valid and return any valid string.

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
Example 2:

Input: s = "a)b(c)d"
Output: "ab(c)d"
Example 3:

Input: s = "))(("
Output: ""
Explanation: An empty string is also valid.
"""
class Solution: 
    def minRemoveToMakeValid(self, s: str) -> str:
        res = []
        cnt = 0 # extra ( parentheses 
        for c in s:
            if c == "(":
                res.append(c)
                cnt += 1
            elif c == ")" and cnt > 0:
                res.append(c)
                cnt -= 1
            elif c != ")":
                res.append(c)
        filtered = []
        for c in res[::-1]:
            if c == "(" and cnt > 0:
                cnt -= 1
            else:
                filtered.append(c)
        return "". join(filtered[::-1])
