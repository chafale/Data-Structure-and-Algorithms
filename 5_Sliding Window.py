from ast import List
from collections import defaultdict
from collections import deque 
from collections import Counter
import heapq
import math

"""
===================================================================================================
 TWO POINTERS AND SLIDING WINDOW PATTERNS
===================================================================================================

    1. Constant window size
    Pick the k elements consecutively from the array
    e.g : nums = [-1, 2, 3 , 3, 4, 5, -1, -2] k = 4

    ** Most frequently asked pattern
    2. Longest subarray / substring where <condition>   
    e.g : Longest subarray where sum of elements is greater than or equal to k
    
            l, r = 0, 0
            max_sum = 0
            max_len = 0
            while r < len(nums):
                max_sum += nums[r]

                # check if the window id invalid
                while max_sum > k:
                    max_sum -= nums[l]
                    l += 1

                # check if the window is valid
                if max_sum <= k: 
                    max_len = max(max_len, r - l + 1)

                r += 1
 
            return max_len

    3. Shortest / Minimum window with < condition >
        Approach :
        1. First find the valid window 
        2. Then try to skrink window size until it's valid

    4. Fourth pattern is the subarray count :
       Number of subarrays with sum equal to target = (Number of subarrays with sum <= target) - (Number of subarrays with sum <= target - 1)

        * Subarrays with sum less than or equal to target include subarrays that have sums less than and equal to the target.
        * Subarrays with sum less than or equal to target - 1 only include those subarrays whose sums are strictly less than the target.
        * So, the difference between these two counts gives you only the subarrays whose sum is exactly equal to the target.
"""




# Problem 1 - Sliding Window Maximum
"""
* * HARD PROBLEM
https://leetcode.com/problems/sliding-window-maximum/description/
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
            if (r - l) + 1 >= k:
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

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
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

# standard pattern solution
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        l, r = 0, 0
        char_freq_cnt = {}
        max_freq_so_far = 0
        substr_len = 0
        while r < len(s):
            char_freq_cnt[s[r]] = char_freq_cnt.get(s[r], 0) + 1
            max_freq_so_far = max(max_freq_so_far, char_freq_cnt[s[r]])

            # check if the window is invalid
            if (r - l + 1) - max_freq_so_far > k:
                char_freq_cnt[s[l]] -= 1
                l += 1

            substr_len = max(substr_len, (r - l + 1))
            r += 1

        return substr_len



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




# 8. Subarray Product Less Than K
"""
https://leetcode.com/problems/subarray-product-less-than-k/
Given an array of integers nums and an integer k, return the number of contiguous subarrays 
where the product of all the elements in the subarray is strictly less than k.

Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
"""
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # using sliding window technique
        l, r = 0, 0
        prod = 1
        count = 0
        while r < len(nums):
            prod = prod * nums[r]
            
            while l < len(nums) and prod >= k:
                prod = prod / nums[l]
                l += 1

            if prod < k:
                count += (r - l + 1)
                
            r += 1

        return count
    

 

# 9. Max Consecutive Ones III
"""
* * Important question
https://leetcode.com/problems/max-consecutive-ones-iii
Given a binary array nums and an integer k, return the maximum number of consecutive 1's in 
the array if you can flip at most k 0's.

Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
                        -----------
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
"""
# Approach : Find the longest subarray with atmost k 0's
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        res = 0

        l, r = 0, 0
        cnt = 0
        while r < len(nums):
            if nums[r] == 0:
                cnt += 1

            # sliding window shrinking condition cnt <= k
            while cnt > k:
                if nums[l] == 0:
                    cnt -= 1
                l += 1

            res = max(res, (r - l) + 1)
            r += 1

        return res




# 10. Maximum Points You Can Obtain from Cards 
"""
https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/description/

There are several cards arranged in a row, and each card has an associated number 
of points. The points are given in the integer array cardPoints.

In one step, you can take one card from the beginning or from the end of the row. 
You have to take exactly k cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array cardPoints and the integer k, return the 
maximum score you can obtain.


Example 1:

Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.
Example 2:

Input: cardPoints = [2,2,2], k = 2
Output: 4
Explanation: Regardless of which two cards you take, your score will always be 4.
Example 3:

Input: cardPoints = [9,7,7,9,7,7,9], k = 7
Output: 55
Explanation: You have to take all the cards. Your score is the sum of points of all cards.

https://www.youtube.com/watch?v=pBWCOCS636U&list=PLgUwDviBIf0q7vrFA_HEWcqRqMpCXzYAL&index=2
"""
class Solution:
    """
      [ 1, 2, 3, 4, 5, 6, 1 ]
       ___________
                 ^        ^
                 |        |
                 l_ptr    r_ptr
    """
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        if k == len(cardPoints):
            return sum(cardPoints)

        l_sum = sum(cardPoints[:k])
        r_sum = 0

        l_ptr, r_ptr = k - 1, len(cardPoints) - 1

        max_sum = l_sum + r_sum

        while l_ptr >= 0:
            l_sum = l_sum - cardPoints[l_ptr]
            l_ptr -= 1

            r_sum = r_sum + cardPoints[r_ptr]
            r_ptr -= 1

            max_sum = max(max_sum, l_sum + r_sum)

        return max_sum




# 11. Fruit Into Baskets 
"""
https://leetcode.com/problems/fruit-into-baskets/description/
You are visiting a farm that has a single row of fruit trees arranged from left to right. 
The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces.

You want to collect as much fruit as possible. However, the owner has some strict rules 
that you must follow:
1. You only have two baskets, and each basket can only hold a single type of fruit. There is no 
   limit on the amount of fruit each basket can hold.
2. Starting from any tree of your choice, you must pick exactly one fruit from every tree (including 
   the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
3. Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

Given the integer array fruits, return the maximum number of fruits you can pick.

Example 1:

Input: fruits = [1,2,1]
Output: 3
Explanation: We can pick from all 3 trees.
Example 2:

Input: fruits = [0,1,2,2]
Output: 3
Explanation: We can pick from trees [1,2,2].
If we had started at the first tree, we would only pick from trees [0,1].
Example 3:

Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: We can pick from trees [2,3,2,2].
If we had started at the first tree, we would only pick from trees [1,2].
"""
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        l, r = 0, 0
        basket_map = {} # hashmap <fruits, frequency>
        max_len = 0

        while r < len(fruits):
            basket_map[fruits[r]] = basket_map.get(fruits[r], 0) + 1

            # check if the sliding window is invalid
            while len(basket_map) > 2:
                basket_map[fruits[l]] -= 1
                if basket_map[fruits[l]] == 0:
                    del basket_map[fruits[l]]
                l += 1

            max_len = max(max_len, r - l + 1)

            r += 1

        return max_len




# 12. Longest Substring with At Most K Distinct Characters 
# solution is same as above
"""
https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/
Given a string s and an integer k, return the length of the longest 
substring of s that contains at most k distinct characters.

Example 1:

Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.
Example 2:

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
"""
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        l, r = 0, 0
        distinct_character_map = {}
        max_len = 0
        while r < len(s):
            distinct_character_map[s[r]] = distinct_character_map.get(s[r], 0) + 1

            # check if the slinding window condition is invalid
            while len(distinct_character_map) > k:
                distinct_character_map[s[l]] -= 1
                if distinct_character_map[s[l]] == 0:
                    del distinct_character_map[s[l]]
                l += 1

            max_len = max(max_len, r - l + 1)

            r += 1

        return max_len




# 13. Number of Substrings Containing All Three Characters 
"""
*** Tricky question might skip while revision
https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/description/
Given a string s consisting only of characters a, b and c.

Return the number of substrings containing at least one occurrence of all these characters a, b and c.

 

Example 1:

Input: s = "abcabc"
Output: 10
Explanation: The substrings containing at least one occurrence of the characters a, b and c are "abc", "abca", "abcab", "abcabc", "bca", "bcab", "bcabc", "cab", "cabc" and "abc" (again). 
Example 2:

Input: s = "aaacb"
Output: 3
Explanation: The substrings containing at least one occurrence of the characters a, b and c are "aaacb", "aacb" and "acb". 
Example 3:

Input: s = "abc"
Output: 1

https://youtu.be/xtqN4qlgr8s?list=PLgUwDviBIf0q7vrFA_HEWcqRqMpCXzYAL&t=420
"""

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        r = 0
        last_seen = {'a': -1, 'b': -1, "c": -1} # last seen of characters a, b, c
        substr_cnt = 0
        while r < len(s):
            last_seen[s[r]] = r

            if last_seen['a'] != -1 and last_seen['b'] != -1 and last_seen['c'] != -1:
                # finding the occurrence of thr min window of `abc` and every character 
                # before it will alway be the substr of `abc`
                substr_cnt += 1 + min(last_seen['a'], last_seen['b'], last_seen['c'])
            r += 1

        return substr_cnt
    



# 14. Binary Subarrays With Sum 
"""
https://leetcode.com/problems/binary-subarrays-with-sum/description/

Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.

A subarray is a contiguous part of the array.

 Example 1:

Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
Example 2:

Input: nums = [0,0,0,0,0], goal = 0
Output: 15

https://youtu.be/XnMdNUkX6VM?list=PLgUwDviBIf0q7vrFA_HEWcqRqMpCXzYAL&t=1024
"""

# Solution 1 : Using prefix sum and counting the number of prefix with sum equal to goal
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        prefix_sum_freq_map = {0: 1}
        res = 0
        curr_sum = 0
        for num in nums:
            curr_sum += num
            if curr_sum - goal in prefix_sum_freq_map:
                res += prefix_sum_freq_map[curr_sum - goal]
            prefix_sum_freq_map[curr_sum] = prefix_sum_freq_map.get(curr_sum, 0) + 1
        return res
    
# Solution 2 : Using sliding window
# Approach : solve using the 4th pattern above 
# Number of subarrays with sum equal to target = (Number of subarrays with sum <= target) - (Number of subarrays with sum <= target - 1)
        # * Subarrays with sum less than or equal to target include subarrays that have sums less than and equal to the target.
        # * Subarrays with sum less than or equal to target - 1 only include those subarrays whose sums are strictly less than the target.
        # * So, the difference between these two counts gives you only the subarrays whose sum is exactly equal to the target.




# 15. Count Number of Nice Subarrays
"""
Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.

Return the number of nice sub-arrays.

Example 1:

Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].
Example 2:

Input: nums = [2,4,6], k = 1
Output: 0
Explanation: There are no odd numbers in the array.
Example 3:

Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
Output: 16
"""
# This problem is same as the problem 14 - Binary Subarrays With Sum 
# just change all even numbers to zero and all odd numbers to one and solve the problem using any method above




