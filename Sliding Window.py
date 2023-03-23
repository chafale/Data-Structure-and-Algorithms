from ast import List
from collections import defaultdict
from collections import deque 
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
# Uses the concept of monotonic queue
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        result = []
        q = deque()

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
