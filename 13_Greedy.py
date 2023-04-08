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
    



# 4. Jump Game 
"""
https://leetcode.com/problems/jump-game
You are given an integer array nums. You are initially positioned at the array's first index, 
and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

Example 1:
Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. 
Its maximum jump length is 0, which makes it impossible to reach the last index.

https://www.youtube.com/watch?v=Yan0cv2cLy8
"""
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal_post = len(nums) - 1

        for i in range(len(nums) -1, -1, -1):
            if i + nums[i] >= goal_post:
                goal_post = i

        return True if goal_post == 0 else False




# 5. Jump Game II
"""
You are given a 0-indexed array of integers nums of length n. 
You are initially positioned at nums[0].

Each element nums[i] represents the maximum length of a forward jump from index i. 
In other words, if you are at nums[i], you can jump to any nums[i + j]

Return the minimum number of jumps to reach nums[n - 1]. 
The test cases are generated such that you can reach nums[n - 1].

Example 1:
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. 
             Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: nums = [2,3,0,1,4]
Output: 2
"""
# we will use the concept of BFS to solve this problem
class Solution:
    def jump(self, nums: List[int]) -> int:
        l = r = 0
        level = 0
        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            level += 1
        
        return level
    



# 6. Gas Station
"""
https://leetcode.com/problems/gas-station/
There are n gas stations along a circular route, where the amount of gas at the ith station 
is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith 
station to its next (i + 1)th station. You begin the journey with an empty tank at one of the 
gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel 
around the circuit once in the clockwise direction, otherwise return -1.

If there exists a solution, it is guaranteed to be unique
"""
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # check whether solution exists
        if sum(gas) < sum(cost):
            return -1
        
        total = 0
        idx = 0
        for i in range(len(gas)):
            total += (gas[i] - cost[i])

            if total < 0:
                total = 0
                idx = i + 1

        return idx
        
