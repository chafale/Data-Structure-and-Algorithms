from ast import List
from collections import defaultdict
from collections import deque 
import heapq
import math


# 1. Push Dominoes
"""
https://leetcode.com/problems/push-dominoes/
You are given a string dominoes representing the initial state where:
dominoes[i] = 'L', if the ith domino has been pushed to the left,
dominoes[i] = 'R', if the ith domino has been pushed to the right, and
dominoes[i] = '.', if the ith domino has not been pushed.
Return a string representing the final state.

Input: dominoes = "RR.L"
Output: "RR.L"
Explanation: The first domino expends no additional force on the second domino.

Input: dominoes = ".L.R...LR..L.."
Output: "LL.RR.LLRRLL.."
"""
# solution : specially with 'R' domino we will check its left is '.' and next to 
# dot is there a 'L' -> if yes: dot will remain as it is otherwise dot will be changed to 'R'

# we will be using queue data structure

class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        dominoes = list(dominoes)
        q = deque()

        for i, d in enumerate(dominoes):
            if d != ".":
                q.append((i, d))

        while q:
            i, d = q.popleft()

            if d == "L":
                if i > 0 and dominoes[i-1] == ".":
                    dominoes[i-1] = "L"
                    q.append((i-1, "L"))
            elif d == "R":
                if i+1 < len(dominoes) and dominoes[i+1] == ".":
                    # check if i+2 index does not have "L"
                    # If R  .  L is the case ; then we pop
                    #    i i+1 i+2
                    if i+2 < len(dominoes) and dominoes[i+2] == "L":
                        # we pop bcoz we don't want to visit domino at index i+2
                        q.popleft()
                    else:
                        dominoes[i+1] = "R"
                        q.append((i+1, "R"))

        return "".join(dominoes)




# 2.  Largest Rectangle in Histogram
"""
Given an array of integers heights representing the histogram's bar height 
where the width of each bar is 1, return the area of the largest rectangle in the histogram.

Link : https://leetcode.com/problems/largest-rectangle-in-histogram/
"""

# Approach : using monotonic increasing stack
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        max_area = 0
        for index, height in enumerate(heights):
            start = index
            while stack and stack[-1][1] > height:
                i, h = stack.pop()
                max_area = max(max_area, h * (index - i))
                start = i
            stack.append([start, height])
        
        while stack:
            i, h = stack.pop()
            max_area = max(max_area, h * (len(heights) - i))
        return max_area
    



# 3. Maximum Frequency Stack
"""
https://leetcode.com/problems/maximum-frequency-stack/
Design a stack-like data structure to push elements to the stack and pop the most frequent element 
from the stack.

Implement the FreqStack class:
1. FreqStack() constructs an empty frequency stack.
2. void push(int val) pushes an integer val onto the top of the stack.
3. int pop() removes and returns the most frequent element in the stack.
    If there is a tie for the most frequent element, the element closest to the stack's top is 
    removed and returned.
"""
