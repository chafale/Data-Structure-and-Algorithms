from ast import List
import sys
import math
from typing import Optional
import heapq

# =======================================================================================================
# ========================================== HEAP PROBLEMS ==============================================
# =======================================================================================================

# 1. Find K Pairs with Smallest Sums
"""
https://leetcode.com/problems/find-k-pairs-with-smallest-sums
You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u, v) which consists of one element from the first array and one element from the 
second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.

Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]
Explanation: The first 3 pairs are returned from the sequence: 
[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

Input: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
Output: [[1,1],[1,1]]
Explanation: The first 2 pairs are returned from the sequence: 
[1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]

Input: nums1 = [1,2], nums2 = [3], k = 3
Output: [[1,3],[2,3]]
Explanation: All possible pairs are returned from the sequence: [1,3],[2,3]
"""
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        m, n = len(nums1), len(nums2)
        heap = [[nums1[0] + nums2[0], (0, 0)]]
        visited = set((0, 0))

        res = []
        while k > 0 and heap:
            _, (i, j) = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])

            if i+1 < m and (i+1, j) not in visited:
                heapq.heappush(heap, [nums1[i+1] + nums2[j], (i+1, j)])
                visited.add((i+1, j))

            if j+1 < n and (i, j+1) not in visited:
                heapq.heappush(heap, [nums1[i] + nums2[j+1], (i, j+1)])
                visited.add((i, j+1))
            k -= 1

        return res
    



# 2. Find Median from Data Stream
"""
https://leetcode.com/problems/find-median-from-data-stream/description/
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
1. void addNum(int num) adds the integer num from the data stream to the data structure.
2. double findMedian() returns the median of all elements so far. Answers within 10-5 of the 
   actual answer will be accepted.

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
""" 
class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        # two heaps, large, small, minheap, maxheap
        # heaps should be equal size
        self.small, self.large = [], []  # maxHeap, minHeap (python default)

    def addNum(self, num: int) -> None:
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -1 * num)

        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        return (-1 * self.small[0] + self.large[0]) / 2




# 1. Kth Largest Element in a Stream
"""
* * Good Problem
Design a class to find the kth largest element in a stream. Note that it is the kth 
largest element in the sorted order, not the kth distinct element.

Implement KthLargest class:
1. KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of 
   integers nums.
2. int add(int val) Appends the integer val to the stream and returns the element representing 
   the kth largest element in the stream.
"""
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        self.k = k

        heapq.heapify(self.heap)

        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)

        while len(self.heap) > self.k:
            heapq.heappop(self.heap)

        return self.heap[0]
    



# 2. Last Stone Weight
"""
https://leetcode.com/problems/last-stone-weight
You are given an array of integers stones where stones[i] is the weight of the ith stone.

We are playing a game with the stones. On each turn, we choose the heaviest two stones and smash 
them together. Suppose the heaviest two stones have weights x and y with x <= y. 
The result of this smash is:
1. If x == y, both stones are destroyed, and
2. If x != y, the stone of weight x is destroyed, and the stone of weight y 
   has new weight y - x.

At the end of the game, there is at most one stone left.

Return the weight of the last remaining stone. If there are no stones left, return 0.
"""
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = [-1 * stone for stone in stones]
        heapq.heapify(heap)
        while heap and len(heap) != 1:
            elem1 = -1 * heapq.heappop(heap)
            elem2 = -1 * heapq.heappop(heap)

            smash = elem1 - elem2
            if smash > 0:
                heapq.heappush(heap, -1 * smash)

        return -1 * heap[-1] if heap else 0
    



# 3. Kth Largest Element in an Array
"""
https://leetcode.com/problems/kth-largest-element-in-an-array/
Given an integer array nums and an integer k, return the kth largest element in the array.
You must solve it in O(n) time complexity.
"""
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = [-1 * num for num in nums]
        heapq.heapify(heap)
        cnt = 0
        while True:
            tmp = heapq.heappop(heap)
            cnt += 1
            if cnt == k:
                return -1 * tmp
            



