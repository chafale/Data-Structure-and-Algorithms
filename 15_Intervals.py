from ast import List
from collections import defaultdict
from collections import Counter
from collections import deque 
from typing import Optional
import heapq
import math
import random


# 1. Insert Interval
"""
* * Good question
https://leetcode.com/problems/insert-interval/
You are given an array of non-overlapping intervals intervals
and intervals is sorted in ascending order by start index.

Insert newInterval into intervals such that intervals is still sorted in 
ascending order by starti and intervals still does not have any overlapping intervals 
(merge overlapping intervals if necessary).

Return intervals after the insertion.

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
"""
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []

        for i in range(len(intervals)):
            # new interval doesn't overlap with current interval and is in front of the current interval
            if newInterval[1] < intervals[i][0]:
                res.append(newInterval)
                return res + intervals[i:]
            # new interval doesn't overlap with current interval and is next of the current interval
            elif newInterval[0] > intervals[i][1]:
                res.append(intervals[i])
            # overlap between current and new_interval
            else:
                newInterval = [
                    min(newInterval[0], intervals[i][0]),
                    max(newInterval[1], intervals[i][1]),
                ]

        # append the newInterval since new interval keeps merging with elements till the end
        # and we didn't add the new-interval to our result
        res.append(newInterval)
        return res
    



# 2. Merge Intervals
"""
https://leetcode.com/problems/merge-intervals/
Given an array of intervals, merge all overlapping intervals, and 
return an array of the non-overlapping intervals that cover all the intervals in the input.

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()

        stack = [intervals[0]]
        for start, end in intervals[1:]:
            if start > stack[-1][1]:
                stack.append([start, end])
            else:
                # just update the end time of interval as interval list is already sorted 
                # -- start need not be updated
                stack[-1][1] = max(stack[-1][1], end)

        return stack
    



# 3. Non-overlapping Intervals
"""
* Good question
https://leetcode.com/problems/non-overlapping-intervals/
Given an array of intervals intervals return the minimum number of intervals you need 
to remove to make the rest of the intervals non-overlapping.

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
"""
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        
        count = 0
        prevIntervalEnd = intervals[0][1]
        for start, end in intervals[1:]:
            if start >= prevIntervalEnd:
                prevIntervalEnd = end
            else:
                count += 1
                prevIntervalEnd = min(prevIntervalEnd, end)

        return count
    



# 4. Meeting Rooms
"""
https://leetcode.com/problems/meeting-rooms/
Given an array of meeting time intervals where intervals[i] = [starti, endi], 
determine if a person could attend all meetings.

Input: intervals = [[0,30],[5,10],[15,20]]
Output: false

Input: intervals = [[7,10],[2,4]]
Output: true
"""
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        if not intervals:
            return True
            
        intervals.sort()

        prevInterval = intervals[0]
        for curr_interval in intervals[1:]:
            # if current intervals starts before previous interval end -- then overlap
            if curr_interval[0] < prevInterval[1]:
                return False
            prevInterval = curr_interval
            
        return True
    



# 5. Meeting Rooms II
"""
https://leetcode.com/problems/meeting-rooms-ii/
Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], 
return the minimum number of conference rooms required.

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2

Input: intervals = [[7,10],[2,4]]
Output: 1
"""
