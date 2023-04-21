from ast import List
import math


# 1. Boats to Save People
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
    



# 2. Jump Game 
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




# 3. Jump Game II
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
    



# 4. Gas Station
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
        



# 5. Two City Scheduling
"""
https://leetcode.com/problems/two-city-scheduling

A company is planning to interview 2n people.

Given the array costs where costs[i] = [aCosti, bCosti], the cost of flying the 
ith person to city a is aCosti, and the cost of flying the ith person to city b is bCosti.

Return the minimum cost to fly every person to a city such that exactly n people 
arrive in each city.

Input: costs = [[10,20],[30,200],[400,50],[30,20]]
Output: 110
Explanation: 
The first person goes to city A for a cost of 10.
The second person goes to city A for a cost of 30.
The third person goes to city B for a cost of 50.
The fourth person goes to city B for a cost of 20.

The total minimum cost is 10 + 30 + 50 + 20 = 110 to have half the people interviewing in each city.
"""
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # calculate how expensive is to send each person to city B
        difference = [(bCost - aCost, aCost, bCost) for aCost, bCost in costs]

        # sort
        difference.sort()

        res = 0
        for i in range(len(difference)):
            # initial half people will go to city B and remaining will go to city A
            if i < len(difference)//2:
                res += difference[i][2] # b cost
            else:
                res += difference[i][1] # a cost

        return res
    



# 6. Eliminate Maximum Number of Monsters
"""
https://leetcode.com/problems/eliminate-maximum-number-of-monsters
You are playing a video game where you are defending your city from a group of n monsters.
You are given a 0-indexed integer array dist of size n, where dist[i] is the initial distance in 
kilometers of the ith monster from the city.

The speed of each monster is given to you in an integer array speed of size n

You have a weapon that, once fully charged, can eliminate a single monster. However, the weapon 
takes one minute to charge.The weapon is fully charged at the very start.

You lose when any monster reaches your city. If a monster reaches the city at the exact moment the 
weapon is fully charged, it counts as a loss, and the game ends before you can use your weapon.

Return the maximum number of monsters that you can eliminate before you lose, or n if you can eliminate
all the monsters before they reach the city.

Input: dist = [1,3,4], speed = [1,1,1]
Output: 3
Explanation:
In the beginning, the distances of the monsters are [1,3,4]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,2,3]. You eliminate the second monster.
After a minute, the distances of the monsters are [X,X,2]. You eliminate the thrid monster.
All 3 monsters can be eliminated.

Input: dist = [1,1,2,3], speed = [1,1,1,1]
Output: 1
Explanation:
In the beginning, the distances of the monsters are [1,1,2,3]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,0,1,2], so you lose.
You can only eliminate 1 monster.

Input: dist = [3,2,4], speed = [5,3,2]
Output: 1
Explanation:
In the beginning, the distances of the monsters are [3,2,4]. You eliminate the first monster.
After a minute, the distances of the monsters are [X,0,2], so you lose.
You can only eliminate 1 monster.
"""
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        # more of the simulation based question

        # calculate the time at which each monster reaches the city
        minReached = []
        for d, s in zip(dist, speed):
            minute = math.ceil(d/s)
            minReached.append(minute)

        # sort the array
        minReached.sort()

        res = 0
        for time in range(len(minReached)):
            # if current time is greater of equal to the moster reaching time then game over
            if time >= minReached[time]:
                return res
            res += 1
        return res
    



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
    



# 8. Partition Labels
"""
https://leetcode.com/problems/partition-labels/
You are given a string s. We want to partition the string into as many parts as 
possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, 
the resultant string should be s.

Return a list of integers representing the size of these parts.

Note : for simple explaination : we just want to create partition such that every char in the 
the partition is only present in that particular partition and no other partion has that char

i. e no overlap of partition characters

Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".

Input: s = "eccbbbbdec"
Output: [10]
Explanation: complete partion 
"""
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        charLastIndex = {}
        for i, char in enumerate(s):
            charLastIndex[char] = i

        size = 0
        end = 0

        res = []
        for i in range(len(s)):
            size += 1
            end = max(end, charLastIndex[s[i]])

            if i == end:
                res.append(size)
                size = 0
        return res




