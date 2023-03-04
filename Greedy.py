from ast import List

# Problem 1 - Merge Intervals
"""
https://leetcode.com/problems/merge-intervals/

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals
Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
"""
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # sort the intervals list
        intervals.sort()

        # create a new merged_intervals list to store the result
        merged_intervals = [intervals[0]]

        for interval in intervals[1:]: # start from 1 index
            # check if there is an overlap
            if interval[0] <= merged_intervals[-1][1]:
                # merge the interval
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
            else:
                # since no overlap just append the interval
                merged_intervals.append(interval)

        return merged_intervals




# Problem 2 - Meeting Rooms II
"""
Given an array of meeting time intervals intervals where 
    intervals[i] = [starti, endi], 
return the minimum number of conference rooms required.

Example 1:

Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2
Example 2:

Input: intervals = [[7,10],[2,4]]
Output: 1

https://www.youtube.com/watch?v=FdzJmTCVyJU
"""
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        pass