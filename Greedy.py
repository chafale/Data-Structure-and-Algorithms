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




# Problem 3 - Boats to Save People
"""
https://leetcode.com/problems/boats-to-save-people
You are given an array people where people[i] is the weight of the ith person, 
and an infinite number of boats where each boat can carry a maximum weight of limit. 
Each boat carries at most two people at the same time, provided the sum of the weight 
of those people is at most limit.

Return the minimum number of boats to carry every given person.

Example 1:
Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)

Example 2:
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)

Example 3:
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)
"""
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        l ,r = 0, len(people) - 1
        no_of_boats = 0
        while l <= r:
            if people[l] + people[r] <= limit:
                no_of_boats += 1
                l += 1
                r -= 1
            else:
                no_of_boats += 1
                r -= 1
        return no_of_boats