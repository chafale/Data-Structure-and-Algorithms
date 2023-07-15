from ast import List
import sys
import math
import bisect
"""
Dynamic Programming :-
It can be applied where there is :-
1. Overlapping sub-problems
2. Optimal substructure

Identification of the DP problem:
1. Count the total number of ways
2. Minimize and Miaximize the output

Memoization : Recursion: (Top-down) when we store value of subproblem in a map/table
Tabulation: (Bottom-up) Start from base case to required problem

Steps for solving any dynamic programming problem:
Step 1: identify the sub-problem, find the base case
Step 2: identify the flow (Go from easiest to complex)
Step 3: Solve the problem recursively

Recursion/Memoization -> Tabulation -> optimize 



* * * * * *
            Note : In Tabulation ==> the for loop will the in opposite dir of recurrence

            if recurrence is going from n-1 -> 0
                for loop will go from 0 -> n-1

                . . . and vice versa
"""

# =============================================================================================
# ======================================= DP in 1D ============================================
# =============================================================================================

# Problem - 1 Climbing Stairs
"""
https://leetcode.com/problems/climbing-stairs/
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        dp = [0 for i in range(n+1)]
        dp[0], dp[1] = 1, 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n] 




# Problem - 2 Frog Jump
"""
https://www.codingninjas.com/codestudio/problems/frog-jump_3621012
Given zero based indexes frog can either jump 1 or 2 steps.
Energy used in jump is given by ht[j] - ht[i]
Min energy to reach the last step.

U are given heights array

Input : [10, 20, 30, 10]
Output : 20
"""
def frogJump(n: int, heights: List[int]) -> int:
    dp = [0 for i in range(n)]
    dp[0] = 0
    for i in range(1, n):
        # last index + energy spent
        s1 = dp[i-1] + abs(heights[i] - heights[i-1])

        # last last index + energy spent
        s2 = float("inf")
        if i > 1:
            s2 = dp[i-2] + abs(heights[i] - heights[i-2])

        dp[i] = min(s1, s2)

    return dp[n-1]

# OPTIMIZATION
# dp[i-1] --> prev
# dp[i-2] --> prev2
# dp[i] --> curr
def frogJump(n: int, heights: List[int]) -> int:
    prev, prev2 = 0, 0
    for i in range(1, n):
        # last dp + energy spent
        s1 = prev + abs(heights[i] - heights[i-1])

        # last last dp + energy spent
        s2 = float("inf")
        if i > 1:
            s2 = prev2 + abs(heights[i] - heights[i-2])
        curr = min(s1, s2)

        prev2 = prev
        prev = curr

    return prev




# Problem - 3 Maximum sum of non-adjacent elements
"""
https://www.codingninjas.com/codestudio/problems/maximum-sum-of-non-adjacent-elements_843261
You are given an array/list of 'N' integers. 
You are supposed to return the maximum sum of the subsequence with the constraint that 
no two elements are adjacent in the given array/list.
Input : a = [2, 4, 1, 9]
Output : 11
"""
# since it's non adjacent then its subsequence
# recurrence
a = []
def f(idx):
    if idx == 0: return a[0]
    if idx < 0: return 0 
    # pick
    p = a[idx] + a[idx - 2]
    # non pick
    np = 0 + a[idx - 1]
    return max(p, np)

def maximumNonAdjacentSum(a):    
    dp = [0 for i in range(len(a))]

    # if idx == 0: return a[0]
    dp[0] = a[0]

    # if idx < 0: return 0 
    neg = 0

    for i in range(1, len(a)):
        # pick
        p = a[i] + (dp[i - 2] if i > 1 else neg)

        # non pick
        np = 0 + dp[i - 1]

        dp[i] = max(p, np)
    return dp[len(a)-1]

# OPTIMIZATION
# dp[i-1] --> prev
# dp[i-2] --> prev2
# dp[i] --> curr
def maximumNonAdjacentSum(a):    
    prev, prev2 = a[0], 0
    neg = 0
    for i in range(1, len(a)):
        # pick
        p = a[i] + (prev2 if i > 1 else neg)
        # non pick
        np = 0 + prev
        curr = max(p, np)

        prev2 = prev
        prev = curr

    return prev




# Problem - 4 House Robber II
"""
https://leetcode.com/problems/house-robber-ii/

You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed. All houses at this 
place are arranged in a circle. That means the first house is the neighbor 
of the last one. Meanwhile, adjacent houses have a security system connected, 
and it will automatically contact the police if two adjacent houses were 
broken into on the same night.

Given an integer array nums representing the amount of money of each house, 
return the maximum amount of money you can rob tonight without alerting the police.

Example 1:
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), 
because they are adjacent houses.

Example 2:
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
"""
# Solution : My answer cannot contain the first and last index at the same Time
# ans 1 = array[0: last -1 ], leave the last element ans solve max sum of non-adj element
# ans 2 = array[1 : last], leaving out the first element
# Ans  = MAX( ans1, ans 2)







# =============================================================================================
# ======================================= DP in 2D ============================================
# =============================================================================================

# Problem - 6 Ninja's Training
"""
https://www.codingninjas.com/codestudio/problems/ninja-s-training_3621003
Ninja is planing this 'N' days-long training schedule. 
Each day, he can perform any one of these three activities. 
(Running, Fighting Practice or Learning New Moves). 
Each activity has some merit points on each day. 
As Ninja has to improve all his skills, he can't do the same activity 
in two consecutive days. 
Can you help Ninja find out the maximum merit points Ninja can earn?
You are given a 2D array of size N*3 *POINTS' with the points corresponding to 
each day and activity. Your task is to calculate the maximum number of merit 
points that Ninja can earn.

Input : points = [[1,2,5], [3,1,1], [3,3,3]]
Output : 11 (5 + 3 + 3)
"""
# recurrence
points = []
def f(day, last):
    if day == 0:
        maxi = 0
        for task in range(3): # bcoz of 3 task
            if task != last:
                maxi = max(maxi, points[0][task])
        return maxi
    
    maxi = 0
    for task in range(3):
        if task != last:
            point = points[day][task] + f(day-1, task)
            maxi = max(point, maxi)

    return maxi 




# Problem - 7 Unique Paths - I
"""
https://leetcode.com/problems/unique-paths/

Find the total number of unique paths from the cell MATRIX[0][0] 
to MATRIX['M'-1]['N'-1].
To traverse in the matrix, you can either move Right or Down at 
each step. 
For example in a given point MATRIX[i][j]you can move to either MATRIX[i+1][j] 
or MATRIX[i][j+1].
"""
def uniquePaths(m, n):
    dp = [[0 for j in range(n)] for i in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]



# Problem - 8 Unique Paths - II
"""
Matrix having obstacle in the grid
"""
def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
    m, n = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for j in range(n)] for i in range(m)]
    for i in range(m):
        if obstacleGrid[i][0] == 1: break
        dp[i][0] = 1

    for j in range(n):
        if obstacleGrid[0][j] == 1: break
        dp[0][j] = 1 

    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] != 1:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]



# Problem - 9 Minimum Path Sum
"""
https://leetcode.com/problems/minimum-path-sum/

Given a m x n grid filled with non-negative numbers, 
find a path from top left to bottom right, which minimizes 
the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.
"""
def minPathSum(self, grid: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0 for j in range(n)] for i in range(m)]

    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = grid[i][0] + dp[i-1][0]

    for j in range(1, n):
        dp[0][j] = grid[0][j] + dp[0][j-1]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]




# Problem - 10 Triangle problem
"""
You are given a triangular array/list 'TRIANGLE'. 
Your task is to return the minimum path sum to reach from the top to the bottom row.

You can move only to the adjacent number of row below each step. (i.e row below and down diagonal)

Example :
1
2, 3
3,6,7
8,9,6,10
For the given triangle array the minimum sum path would be 1->2->3->8. 
Hence the answer would be 14.
"""
# Approach to solve this problem is to start from the bottom row and build ur answer way up




# Problem - 11 Minimum Falling Path Sum
"""
https://leetcode.com/problems/minimum-falling-path-sum/
You have been given an N*M matrix filled with integer numbers, 
find the maximum sum that can be obtained from a path starting from any cell 
in the first row to any cell in the last row.

From a cell in a row, you can move to another cell directly below that row, 
or diagonally below left or right. 
"""
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        row, col = len(matrix), len(matrix[0])

        dp = [[0 for _ in range(col)] for _ in range(row)]

        for c in range(col):
            dp[row-1][c] =  matrix[row-1][c]

        for r in range(row-2, -1, -1):
            for c in range(col):
                bt_lt = dp[r+1][c-1] if c-1 >= 0 else math.inf
                bt = dp[r+1][c]
                bt_rt = dp[r+1][c+1] if c+1 < col else math.inf

                dp[r][c] = matrix[r][c] + min(bt_lt, bt, bt_rt)

        return min(dp[0])

# Optimization
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        row, col = len(matrix), len(matrix[0])

        curr = [0 for _ in range(col)]
        prev = [matrix[row-1][c] for c in range(col)]

        for r in range(row-2, -1, -1):
            for c in range(col):
                bt_lt = prev[c-1] if c-1 >= 0 else math.inf
                bt = prev[c]
                bt_rt = prev[c+1] if c+1 < col else math.inf

                curr[c] = matrix[r][c] + min(bt_lt, bt, bt_rt)

            prev = curr.copy()
        return min(prev)





# Problem - 12 Cherry Pickup II
"""
skipping for now
"""




# =============================================================================================
# ============================ DP using subseq/subsets with target ============================
# =============================================================================================

# Problem - 13 Subset Sum Equals to Target
"""
* * Note: its subset, not subarray and subsets == subsequences

You are given an array 'ARR' of 'N' positive integers and an integer 'K'. 
Your task is to check if there exists a subset in 'ARR' with a sum equal to 'K'.

Note: Return true if there exists a subset with sum equal to 'K'. Otherwise, return false.
"""
# ============================
# Note : there is a slight change in DP while generating recursion function
#        In dp while generating recursion we start from index "n-1" to "0"
#        This is bcoz in dp we go bottom-up 
#        * * * * * * * * * * 
arr = []
def f(idx, target):
    if target == 0:
        return True
    if idx == 0:
        return arr[0] == target

    # not pick
    not_pick = f(idx - 1, target) # idx - 1 bcoz we are moving from "n-1" to 0

    # pick
    pick = False
    if target >= arr[idx]:
        pick = f(idx-1, target - arr[idx])

    return pick or not_pick # if any of them is true
# ============================
def subset_sum(arr, target):
    n = len(arr)

    # create a dp array
    dp = [[-1 for _ in range(target + 1)] for _ in range(n)]

    # if target == 0
    for i in range(n):
        dp[i][0] = True

    # if idx == 0 and arr[0] == target : then True
    dp[0][arr[0]] = True

    for idx in range(1, n): # 0 is already computed therefore start from 1
        for _target in range(1, target + 1): # target =0 is already computed therefore start from 1
            
            # not pick
            not_pick = dp[i-1][_target]

            # pick
            pick = False
            if _target >= arr[idx]:
                pick = dp[idx-1][_target - arr[idx]]
            
            dp[idx][target] = pick or not_pick

    return dp[n-1][target]




# Problem - 14 Partition Equal Subset Sum
"""
https://leetcode.com/problems/partition-equal-subset-sum/
Given an integer array nums, return true if you can partition the array 
into two subsets such that the sum of the elements in both subsets is 
equal or false otherwise.
"""
# Solution:
# 1. sum(arr) is even check if we can find subset having target = sum(arr)/2
#    if this is possible return True, otherwise False
# 2. sum(arr) is odd ==> return False  
# 
# Think of it as s1 + s2 = S      , where s1 and s2 are subsets
# and s1 = s2 so s1 = S/2 





# Problem - 15 Partition A Set Into Two Subsets With Minimum Absolute Sum Difference
"""
Good question

https://youtu.be/GS_OqZb2CWc?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY&t=714
"""
# Intution :
# we need to find abs| s1 - s2 | to be minimum
#  
# In problem 13 the last row determines whether target: from 0 to k
# is subset_sum possible for the target or not
# 
# so for the last row we will pick all true target values 
#
# we can find s2 from last_row - sum(arr)
# and then find the pair s1(which is last row), s2(sum - s1) which gives min difference 
# target value will be sum(arr) 





# Problem - 16 Counts Subsets with Sum K
"""
Count the number of subsets in the array having target sum = k.
"""
# ============================
arr = []
def f(idx, target):
    if target < 0:
        return 0

    if idx == len(arr):
        if target == 0:
            return 1
        return 0

    # not pick
    not_pick = f(idx + 1, target)

    # pick
    pick = 0
    if arr[idx] <= target:
        pick = f(idx + 1, target - arr[idx])

    return pick + not_pick
# ============================
def count_subset_sum(array, k): # here k is target
    n = len(arr)

    dp = [[0 for _ in range(k + 1)] for _ in range(n)]

    # target == 0: 1
    for r in range(n):
        dp[r][0] = 1

    # idx == 0 and target == arr[0] : 1
    dp[0][arr[0]] = 1

    for idx in range(1, n):
        for target in range(1, k + 1):
            not_pick = dp[idx + 1][target]
            pick = 0
            if arr[idx] <= target:
                pick = dp[idx + 1][target - arr[idx]]

            dp[idx][target] = pick + not_pick

    return dp[n-1][target]




# Problem - 17 Count Partitions With Given Difference
"""
Give arr, and a difference d. Find the number of ways in which we 
can partition arr into two subsets s1 & s2 such that :
s1 - s2 = d and s1 > s2, where s1,s2 are sum of two subsets.
"""
# Solution :  
# s1 + s2 = total and given s1 - s2 = d
# therefore we can say s2 = (tot - d) / 2
# 
# some cases to consider tot - d cannot be neg so tot - d > 0
# and (tot - d ) must be even
# rest solve the problem using subset sum 




# Problem 18 - Target Sum
"""
https://leetcode.com/problems/target-sum/
You are given an integer array nums and an integer target.
You want to build an expression out of nums by adding one of the symbols 
'+' and '-' before each integer in nums and then concatenate all the integers.

Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
"""
# u can solve this problem using  Problem 17 - Count Partitions With Given Difference




# =============================================================================================
# ============================ FAMOUS DP PROBLEMS =============================================
# =============================================================================================


# =============================================================================================
#                                     DP ON SUBSEQUENCES                        
# =============================================================================================

# Problem - 19 0/1 Knapsack or Binary Knapsack
"""
Given weights and values of N items, put these items in a knapsack of capacity W 
to get the maximum total value in the knapsack.

Note: weights can be chosen only once
"""
W = 10
wt = [1, 2, 4, 5]
profit = [5, 4, 8, 6]

# ============================
def f(idx, W):
    if idx == 0:
        if wt[0] <= W:
            return profit[0]
        else: return 0

    # pick
    pick = -math.inf
    if wt[idx] <= W:
        pick = profit[idx] + f(idx - 1, W - wt[idx])

    # not_pick
    not_pick = 0 + f(idx - 1, W)

    return max(pick, not_pick)
# ============================
def binary_knapsack(wt, profit, W):
    n = len(wt)

    dp = [[0 for _ in range(W+1)]for _ in range(n)]

    # idx == 0: wt[0] <= W: profit[0]
    for w in range(wt[0], W+1):
        dp[0][w] = profit[0]

    for idx in range(1, n):
        for wt in range(1, W+1):
            # pick
            pick = -math.inf
            if wt[idx] <= W:
                pick = profit[idx] + dp[idx - 1][W - wt[idx]]

            # not_pick
            not_pick = 0 + dp[idx - 1][W]

            dp[idx][wt] = max(pick, not_pick)

    return dp[n-1][W]




# Problem 20 - Minimum Coins
"""
Find the min number of coins to reach the target sum.
u can pick a certain coin infinite number of times

Input: coins[] = {25, 10, 5}, V = 30
Output: Minimum 2 coins required We can use one coin of 25 cents and one of 5 cents 

Input: coins[] = {9, 6, 5, 1}, V = 11
Output: Minimum 2 coins required We can use one coin of 6 cents and 1 coin of 5 cents
"""
coins = []
def f(idx, target):
    if idx == 0:
        if target % coins[0] == 0:
            return target // coins[0]
        else:
            return math.inf
 
    # pick 
    pick = math.inf
    if coins[idx] <= target:
        pick = 1 + f(idx, target - coins[idx])

    # not pick
    not_pick = 0 + f(idx - 1, target)

    return min(pick, not_pick)





# Problem 21 - Coin Change (or) Target Sum
"""
https://leetcode.com/problems/coin-change
You are given an integer array coins representing coins of different denominations 
and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. 
If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Input: coins = [1,2,5], amount = 11
Output: 3
"""
coins = [1,2,5]
amount = 11

def f(idx, amount):
    if idx == 0:
        if amount % coins[idx] == 0:
            return amount / coins[idx]
        else:
            return math.inf

    # pick
    pick = math.inf
    if coins[idx] >= amount:
        pick = 1 + f(idx, amount - coins[idx])

    # not pick
    not_pick = 0 + f(idx + 1, amount)

    return min(pick, not_pick)

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)

        dp = [[math.inf for _ in range(amount + 1)] for _ in range(n)]

        for i in range(n):
            dp[i][0] = 0

        for amt in range(amount + 1):
            if amt % coins[0] == 0:
                dp[0][amt] = amt // coins[0]

        for idx in range(1, n):
            for target in range(1, amount + 1):
                pick = math.inf
                if coins[idx] <= target:
                    pick = 1 + dp[idx][target - coins[idx]]

                # not pick
                not_pick = 0 + dp[idx - 1][target]

                dp[idx][target] = min(pick, not_pick)

        return dp[n-1][amount] if dp[n-1][amount] != math.inf else -1





# Problem 21 - Coin Change 2
"""
https://leetcode.com/problems/coin-change-ii/
You are given an integer array coins representing coins of different denominations 
and an integer amount representing a total amount of money.

Return the number of combinations that make up that amount. 

If that amount of money cannot be made up by any combination of the coins, return 0.
"""
# finding the number of ways to reach the target
def f(idx, target):
    if idx == 0:
        if target % coins[0] == 0:
            return 1
        else:
            return 0
        
    # pick
    pick = 0
    if coins[idx] <= target:
        pick = f(idx, target - coins[idx])

    # not pick
    not_pick = f(idx - 1, target)

    return pick + not_pick
 
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)

        dp = [[0 for _ in range(amount + 1)] for _ in range(n)]

        for amt in range(amount + 1):
            if amt % coins[0] == 0:
                dp[0][amt] = 1

        for idx in range(1, n):
            for amt in range(amount + 1):
                pick = 0
                if coins[idx] <= amt:
                    pick = dp[idx][amt - coins[idx]]

                not_pick = dp[idx-1][amt]

                dp[idx][amt] = pick + not_pick

        print(dp)

        return dp[n-1][amount]





# Problem 22 - Unbounded Binary Knapsack
"""
In this problem u can choose knapsack wt's infinite number of times (no bound on number of wts
you can pick, <In 0/1 Knapsack u can pick a certain wts only once>

U need to maximize the profit.

wt = [2, 4, 5]
val = [5, 11, 13]

W = 10
"""
wt = [2, 4, 5]
profit = [5, 11, 13]
Knapsack_W = 10
def f(idx, W):
    # base case
    if idx == 0:
        if W % wt[0] == 0:
            return profit[0] * (W // wt[0])
        else:
            return 0

    # pick
    pick = -math.inf
    if wt[idx] <= W:
        pick = profit[idx] + f(idx, W - wt[idx])

    # not pick
    not_pick = 0 + f(idx - 1, W)

    return max(pick, not_pick)




# Problem 23 - Rod Cutting Problem
"""
Given a rod of length n  and array prices of length n denoting the cost of pieces 
of the rod of length 1 to n, 
find the maximum amount that can be made if the rod is cut up optimally.

For example, if the length of the rod is 8 and the values of different pieces are given 
as the following, then the maximum obtainable value is 22 
(by cutting in two pieces of lengths 2 and 6) 

length   | 1   2   3   4   5   6   7   8  
--------------------------------------------
price    | 1   5   8   9  10  17  17  20

And if the prices are as follows, then the maximum obtainable value is 24 
(by cutting in eight pieces of length 1) 

length   | 1   2   3   4   5   6   7   8  
--------------------------------------------
price    | 3   5   8   9  10  17  17  20


Input 1:
n = 8, prices[] = [1, 3, 4, 5, 7, 9, 10, 11]
Output 1:
12
"""
# Unbounded Binary knapsack problem
# Approach :
# we will try to pick length [1, 2, 3, 4, 5] and sum them up to len of rod
# to maximize the profits  
n = 8
prices = [1, 3, 4, 5, 7, 9, 10, 11]
def f(idx, rod_Length):
    if idx == 0:
        return rod_Length * prices[0]

    curr_rod_len = idx + 1
    # pick
    pick = -math.inf
    if curr_rod_len <= rod_Length:
        pick = prices[idx] + f(idx, rod_Length - curr_rod_len)
    #not pick
    not_pick = 0 + f(idx - 1, rod_Length)

    
    


# =============================================================================================
#                                     DP ON SRINGS                        
# =============================================================================================

# Problem 24 - Longest Common Subsequence
"""
https://leetcode.com/problems/longest-common-subsequence/
Given two strings text1 and text2, return the length of their longest common subsequence. 
If there is no common subsequence, return 0.
"""
# Approach
# if s[i] == s[j]:
#    dp[i][j] = 1 + dp[i-1][j-1]
# else:
#    dp[i][j] = max(dp[i-1][j], dp[i][j-1]) 

# ===============++===================+=====+=====+===+===+===
# for two string we use two index
str1, str2 = "", ""
def f(idx1, idx2):
    # base case:
    if idx1 < 0 or idx2 < 0:
        return 0
    if str1[idx1] == str2[idx2]:
        return 1 + f(idx1 - 1, idx2 - 1)
    
    return max(f(idx1-1, idx2), f(idx1, idx2-1))
# ===============++===================+=====+=====+===+===+===
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m+1)]

        # if idx1 < 0 or idx2 < 0:
        # here to accomodate the above base case we will shift the index by 1
        # so 0 becomes 1 and so on

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]





# Problem 25 - Longest Common Substring (LCS)
"""
* * Good problem
https://youtu.be/_wP9mWNPL5w?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY&t=365
"""
# Approach :
# if s1[i] == s2[j]:
#    dp[i][j] = 1 + dp[i-1][j-1]
# else:
#    dp[i][j] = 0
# 
# return the max value in the whole dp matrix
#  





# Problem 26 - Longest Palindromic Subsequence (LPS)
"""
https://leetcode.com/problems/longest-palindromic-subsequence/
Input: s = "bbbab"
Output: 4

"""
# Approach : 
#  1. given string s1 = s
#  2. create string s2 = reverse(s) 
#  3. apply LCS on s1 and s2 
# this is valid bcoz for a string s to be palindrome it should be read same from 
# forward and backwards 
# therefore we can generate subsequences using this intution





# Problem 27 - Minimum Insertions to Make String Palindrome
"""
https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/

Example 1:
Input: s = "zzazz"
Output: 0
Explanation: The string "zzazz" is already palindrome we do not need any insertions.

Example 2:
Input: s = "mbadm"
Output: 2
Explanation: String can be "mbdadbm" or "mdbabdm".

Example 3:
Input: s = "leetcode"
Output: 5
Explanation: Inserting 5 characters the string becomes "leetcodocteel".
"""
# Approach : 
# Main intution is to keep the palindromic portion in a string intact
# 1. find the longest palindromic sunsequence (lps) 
# 2. Ans = len(string) - LPS 




# ===============++===================+=====+=====+===+===+==============
#                  DP STRING MATCHING
# ===============++===================+=====+=====+===+===+==============

# Problem 28 - Distinct Subsequences
"""
https://leetcode.com/problems/distinct-subsequences/
Given two strings s and t, return the number of distinct 
subsequences of s which equals t.

Input: s = "babgbag", t = "bag"
Output: 5
Explanation:
As shown below, there are 5 ways you can generate "bag" from s.
[ba]b[g]bag
[ba]bgba[g]
[b]abgb[ag]
ba[b]gb[ag]
babg[bag]
"""
# for s1 -> i and t -> j
s = "babgbag"
t = "bag"
def f(i, j):
    # base case
    if j < 0: return 1
    if i < 0: return 0

    # if both str char is match
    if s[i] == j[j]:
        return f(i-1, j-1) + f(i-1, j) # s = babgba->(g),  t= ba->(g)
    # both str char does not match
    else:
        return f(i-1, j)

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        pass





# Problem 29 - Edit Distance
"""
https://leetcode.com/problems/edit-distance/
Given two strings word1 and word2, return the minimum number of operations 
required to convert word1 to word2.
You have the following three operations permitted on a word:
* Insert a character
* Delete a character
* Replace a character

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')
"""
def f(i, j):
    # Base case
    if i < 0:
        return j + 1    # very imp -- this many no. of insertion required
    if j < 0:
        return i + 1    # very imp -- this many no. of deletion required

    # when char in string are match
    if s[i] == s[j]:
        return 0 + f(i-1, j-1)    # zero operation + recursion

    #else:
    insertion = 1 + f(i, j-1)
    deletion = 1 + f(i-1, j)
    updation = 1 + f(i-1, j-1)

    return min(insertion, deletion, updation)




# Problem 30 - Wildcard Matching
"""
https://leetcode.com/problems/wildcard-matching/
Given an input string (s) and a pattern (p), implement wildcard pattern 
matching with support for '?' and '*' where:
'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

Input: s = "aa", p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
Example 2:

Input: s = "aa", p = "*"
Output: true
Explanation: '*' matches any sequence.
Example 3:

Input: s = "cb", p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
"""
s = "aa"
p = "*"
def f(i, j):
    # base case
    # both s and p have exhausted
    if i < 0 and j < 0:
        return True

    if i >= 0 and j < 0:
        return False # bcoz u don't have any string left to compare
    
    # s is exhausted and p is left
    if i < 0 and j >= 0:
        # in this case p will match s if all p left are "*" 
        # example s = "ab", p="*****ab"
        while j >= 0:
            if p[j] != "*":
                return False
            j -= 1
        return True

    if s[i] == p[j] or p[j] == "?":
        return f(i-1, j-1)
    elif p[j] == "*":
        return f(i-1, j) or f(i, j-1)

    return False





# =============================================================================================
#                                     DP ON STOCKS                        
# =============================================================================================
# Problem 31 - Best Time to Buy and Sell Stock
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock 
and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. 
If you cannot achieve any profit, return 0.

Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
"""
# Approach : uisng two pointers l, r 
# l is min we have seen so far
# r is max so far
# max_profit
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





# Problem 32 - Best Time to Buy and Sell Stock II
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

* * Note in above problem u can buy and sell only once
* * here u can buy and cell multiple of times

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock. 
You can only hold at most one share of the stock at any time.
However, you can buy it then immediately sell it on the same day.

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
"""
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total_profits = 0
        for i in range(1, len(prices)):
            profit = prices[i] - prices[i-1]
            if profit > 0:
                total_profits += profit

        return total_profits










# ===================== Pre-requisite for the below all problems =====================
# this is the basic recursion for buy and sell stock with infinite transaction

"""
* * Note here in recursion we are moving forward i.e from idx 0 -> n
"""
prices = []
def f(idx, canBuy):
    if idx == len(prices):
        return 0

    profit = 0

    if canBuy:
        # buy stock at current day idx
        buy = -prices[idx] + f(idx + 1, 0)  
        # negative prices bcoz we are buying   and f(.., 0) 
        # becoz we cant buy after we buy this

        # buy in future
        not_buy = 0 + f(idx + 1, 1)

        profit = max(buy, not_buy)

    # if u can't buy that means u have to sell the stock
    else:
        sell = prices[idx] + f(idx + 1, 1)
        not_sell = 0 + f(idx + 1, 0)

        profit = max(sell, not_sell)

    return profit
# ===================== Pre-requisite for the Problem 32 =============================


# Problem 32 - Best Time to Buy and Sell Stock III
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

This is with at most two transactions.   |----- only two
"""

# since for this problem we can only perform 2 trasactions we will have a cap

def f(idx, canBuy, trans_cap):
    if idx == len(prices):
        return 0

    # new cap case added
    if trans_cap == 0:
        return 0

    profit = 0

    if canBuy:
        # buy stock at current day idx
        buy = -prices[idx] + f(idx + 1, 0, trans_cap)   # cap is not reduced bcoz the txn is not complete    

        # buy in future
        not_buy = 0 + f(idx + 1, 1, trans_cap)

        profit = max(buy, not_buy)

    # if u can't buy that means u have to sell the stock
    else:
        sell = prices[idx] + f(idx + 1, 1, trans_cap - 1)
        not_sell = 0 + f(idx + 1, 0, trans_cap)

        profit = max(sell, not_sell)

    return profit


# More optimized approach

# Intitution :
# we can express atmost 2 transaction as => B S B S
# so B : 0, S : 1, B : 2, S : 3
# we are representing Buy & Sell using transaction number   == so by comparing previous problems  
# cap is convert to transaction numbers
def f(idx, transaction):
    if idx == 0 or transaction == 4: 
        return 0

    # if transaction number is even then its buy otherwise its a sell transaction
    if transaction % 2 == 0:  
        buy = -prices[idx] + f(idx + 1, transaction + 1)       
        not_buy = 0 + f(idx + 1, 1, transaction)

        profit = max(buy, not_buy)

    else:
        sell = prices[idx] + f(idx + 1, transaction + 1)
        not_sell = 0 + f(idx + 1, 0, transaction)

        profit = max(sell, not_sell)

    return profit





# Problem 33 - Best Time to Buy and Sell Stock IV
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
This problem is with atmost k transactions
"""
# Solution : 
# Its the same problem as above just change the value 4 to 2 * k 

# Just copy and pasting from above

k = 4

def f(idx, transaction):
    if idx == 0 or transaction == 2 * k:   # (2 * k) bcoz if we have 2 txn we have total 4 buy and sell
        return 0

    # if transaction number is even then its buy otherwise its a sell transaction
    if transaction % 2 == 0:  
        buy = -prices[idx] + f(idx + 1, 0, transaction + 1)       
        not_buy = 0 + f(idx + 1, 1, transaction)

        profit = max(buy, not_buy)

    else:
        sell = prices[idx] + f(idx + 1, 1, transaction + 1)
        not_sell = 0 + f(idx + 1, 0, transaction)

        profit = max(sell, not_sell)

    return profit





# Problem 34 - Best Time to Buy and Sell Stock with Cooldown
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

* * After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
"""
# Solution : small change in the code of infinite transaction
# only one condition to add i.e. we can only buy after a gap of one period after sell
prices = []
def f(idx, canBuy):
    if idx >= len(prices):
        return 0
    profit = 0
    if canBuy:
        buy = -prices[idx] + f(idx + 1, 0)  
        not_buy = 0 + f(idx + 1, 1)
        profit = max(buy, not_buy)

    else:
        sell = prices[idx] + f(idx + 2, 1)    # minor change is here instead of i + 1 ==> we have i + 2
        not_sell = 0 + f(idx + 1, 0)
        profit = max(sell, not_sell)

    return profit




# Problem 35 - Best Time to Buy and Sell Stock with Transaction Fee
"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
"""
# Solution : small change in the code of infinite transaction 
# just add -neg transaction fee while selling 
prices = []
transaction_fee = 2
def f(idx, canBuy):
    if idx == len(prices):
        return 0
    profit = 0
    if canBuy:
        buy = -prices[idx] + f(idx + 1, 0)  
        not_buy = 0 + f(idx + 1, 1)
        profit = max(buy, not_buy)

    else:
        sell = prices[idx] - transaction_fee + f(idx + 1, 1) # transaction fee added
        not_sell = 0 + f(idx + 1, 0)
        profit = max(sell, not_sell)

    return profit





# =============================================================================================
#                                DP ON INCREASING SUBSEQUENCES                        
# =============================================================================================

# Problem 36 - Longest Increasing Subsequence (LIS)
"""
https://leetcode.com/problems/longest-increasing-subsequence/
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
"""
# Two approach to solve this problem
# Approach 1 : using resursion
# Approach 2 : Better approach : But requires additional intution
"""
https://youtu.be/IFfYfonAFGc?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY&t=371
"""
"""
Approach 2 code:
pseudo code:
    dp = [ all ones ] * len(nums)
    nums = [10,9,2,5,3,7,101,18]

    for (idx = 0 -> n-1):
        for (prev = 0 -> idx - 1):
            if nums[prev] < nums[idx]:
                dp[idx] = max(dp[idx], 1 + dp[prev])

                # this is for printing
                hash[i] = j

    return max(dp) 


***** This solution will be required if u want to trace back the LIS
"""
# Most better approach to solve LIS
# Approach 3 : BINARY SEARCH APPROACH
"""
https://youtu.be/on2hvxBXJH4?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY&t=556

Note :
This function only returns the correct len 
the sub=[] is not correct
"""
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for num in nums:
            i = bisect.bisect_left(sub, num)

            # If num is greater than any element in sub
            if i == len(sub):
                sub.append(num)
            
            # Otherwise, replace the first element in sub greater than or equal to num
            else:
                sub[i] = num

        return len(sub)





# Problem 37 - Largest Divisible Subset
"""
https://leetcode.com/problems/largest-divisible-subset/

Given a set of distinct positive integers nums, 
return the largest subset answer such that every pair 
(answer[i], answer[j]) of elements in this subset satisfies:
1. answer[i] % answer[j] == 0, or
2. answer[j] % answer[i] == 0
If there are multiple solutions, return any of them.

Input: nums = [1,2,3]
Output: [1,2]
Explanation: [1,3] is also accepted.
"""
# Approach : 
# 1. we will sort the array after sorting --> the problem will boils down to
#    longest divisible subsequence (if nums are divisible and increasing add them to ans) 
"""
pseudo code:

    dp = [ 1 ] * len(nums)

    nums = nums.sort()

    for (idx = 0 -> n-1):
        for (prev = 0 -> idx):
            if nums[idx] % nums[prev]:                   --> divisibility check
                dp[idx] = max(dp[idx], 1 + dp[prev])

                # this is for printing
                hash[i] = j

    return max(dp) 


***** This solution will be required if u want to trace back the LIS
"""
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:

        hash = {}

        dp = [1] * len(nums)

        for idx in range(len(nums)):
            hash[idx] = idx
            for prev in range(0, idx):
                if nums[idx] % nums[prev] == 0:
                    if dp[idx] < 1 + dp[prev]:
                        dp[idx] = 1 + dp[prev]
                        hash[idx] = prev

        max_, maxi = 0, 0
        for i, elem in enumerate(dp):
            if elem > max_:
                max_ = elem
                maxi = i

        last_idx = maxi
        res = []
        while hash[last_idx] != last_idx:
            res.append(hash[last_idx])
            last_idx = hash[last_idx]

        return res
        




# Problem 38 - Longest String Chain
"""
https://leetcode.com/problems/longest-string-chain/
You are given an array of words.
wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA.
Return the length of the longest possible word chain with words chosen from the given list of words.
Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
"""
# Solve this problem using longest increasing subsequence
class Solution:
    def compareString(self, s1, s2):
        if len(s2) > len(s1):
            s1, s2 = s2, s1

        if len(s1) != len(s2) + 1:
            return False

        i, j = 0, 0
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                j += 1
            i += 1
        
        if i == len(s1) and j == len(s2):
            return True
        elif i < len(s1) and i + 1 == len(s1):
            return True

        return False

    def longestStrChain(self, words: List[str]) -> int:
        words.sort(key=lambda x : len(x))
        dp = [1] * len(words)
        for idx in range(len(words)):
            for prev in range(0, idx):
                if self.compareString(words[idx], words[prev]):
                    dp[idx] = max(dp[idx], 1 + dp[prev])
                    
        return max(dp)





# Problem 39 - Longest Bitonic Subsequence
"""
Bitonic : sequence which increases and then decreases (or) just increasing (or) just decreasing
"""
# create a two dp tables and apply LIS two times 
# first apply from start to end - dp1 - forward LIS
# second apply from end to start - dp2 - reverse LIS (do not reverse the array) just apply LIS from back
# find the ans by dp1[i] + dp2[i] - 1 (-1 bcoz same idx element is counted twice) 
# return max(ans) 






# Problem 40 - Number of Longest Increasing Subsequences
"""
https://leetcode.com/problems/number-of-longest-increasing-subsequence/
Given an integer array nums, return the number of longest increasing subsequences.
Input: nums = [1,3,5,4,7]
Output: 2
Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].
"""
# have another count array to count the number of LIS and increase the count[i] when we get 
# same LIS length from prev elements 
"""
pseudo code:
    nums = [10,9,2,5,3,7,101,18]

    dp = [1] * len(nums)
    count = [1] * len(nums)

    for (idx = 0 -> n-1):
        for (prev = 0 -> idx - 1):
            if nums[prev] < nums[idx] and dp[idx] > 1 + dp[prev]:
                dp[idx] = max(dp[idx], 1 + dp[prev])
            elif nums[prev] < nums[idx] and dp[idx] == 1 + dp[prev]:
                count[idx] += count[prev]

    return max(dp) 
"""
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        count = [1] * len(nums)
        for idx in range(len(nums)):
            for prev in range(idx):
                if nums[idx] > nums[prev] and dp[idx] < 1 + dp[prev]:
                    dp[idx] = dp[idx]
                    # inherit
                    count[idx] = count[prev]
                elif nums[idx] > nums[prev] and dp[idx] == 1 + dp[prev]:
                    # increment
                    count[idx] += count[prev]

        max_lis = max(dp)
        res = 0
        for i in range(len(nums)):
            if dp[i] == max_lis:
                res += count[i]
        return res




# =============================================================================================
#                                       PARTITION DP                        
# =============================================================================================
"""
* * CASES IN WHICH PARTITION DP IS USED
==> When u are given a problem to solve in certain pattern
    e.g mathmatic calculation : 1 + 2 + 3 x 5 
        so get different answers for (1 + 2 + 3) x 5 and (1 + 2) + (3 x 5)

    so when u get a problem where if u solve in one way u get a particular ans and if u
    solve in a different particular way u get different ans 

So in partition dp u will be given an array
        --------------------------------
        |                   |           |
        --------------------------------
       i (start pt)    p (patition pt)    j (end point)
       
       u will have to solve from i -> p partion and p+1 -> j partition

Rules solving partition dp:
1. Start with the entire block
2. Try all partitions -> run a loop to try all partitions
3. return the best possible two partitions
"""

# Problem 41 - Matrix Chain Multiplication
"""
Give dimentions of matrixs, A = [10, 20, 30, 40]

pseudo code:
def f(i, j):
    if i == j:
        return 0

    mini = -inf
    for k = i -> j-1:
        operations = (A[i-1] * A[k] * A[j]) + f(i, k) + f(k+1, j)

        mini = min(mini, operations)

    return mini
"""




# Problem 42 - Minimum Cost to Cut a Stick
"""
https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
Given an integer array cuts where cuts[i] denotes a position you should perform a cut at.
Return the minimum total cost of the cuts.

Input: n = 7, cuts = [1,3,4,5]
Output:
Explanation: Using cuts order = [1, 3, 4, 5] as in the input leads to the following scenario:
The first cut is done to a rod of length 7 so the cost is 7. 
The second cut is done to a rod of length 6 (i.e. the second part of the first cut), 
the third is done to a rod of length 4 
and the last cut is to a rod of length 3. 
The total cost is 7 + 6 + 4 + 3 = 20.

Rearranging the cuts to be [3, 5, 1, 4] for example 
will lead to a scenario with total cost = 16 (as shown in the example 
photo 7 + 4 + 3 + 2 = 16).
"""
# Solution :
# Top- down approach easy and do-able
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        dp = {}
        def recursion(l ,r):
            if (r - l) == 1:
                return 0

            if (l, r) in dp:
                return dp[(l, r)]

            cost = []
            for cutIdx in cuts:
                if l < cutIdx < r:
                    cost.append((r-l) + recursion(l, cutIdx) + recursion(cutIdx, r))

            dp[(l, r)] = min(cost) if cost else 0
            
            return dp[(l, r)]

        return recursion(0, n)

# Bottom - up approach
"""
pseudocode:

add 0 and len(stick) at start and end of cuts array for calculation of length

        --------------------------------
    0   |                   |           |    len(stick)
        --------------------------------
       i (start pt)    p (patition pt)    j (end point)

       
cuts.sort()

def f(i, j):
    if i > j:
        return 0

    min_cost = inf
    for partition = i -> j:
        cost = cuts[j + 1] - cuts[i - 1]
                + f(i, partition - 1)
                + f(partition + 1, j)
        
        min_cost = min(min_cost, cost)

    return min_cost
"""
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        c = len(cuts)

        cuts.sort()
        # add 0 to left and len(cuts) to right
        cuts.insert(0, 0)
        cuts.append(n)
        
        print(cuts)

        dp = [[0 for _ in range(c + 2)] for _ in range(c + 2)]

        # consider the following array     0 ] 1, 3, 4, 5 [ 7 
        # in recurrence i is from 1 -> n and j is from n -> 1   where n = len(cuts) 
        # however in tabulation i and j direction will be reversed
        for i in range(c, 0, -1):
            for j in range(1, c+1):
                if i > j:
                    continue
                min_cost = math.inf
                for partition in range(i, j+1):
                    cost = cuts[j + 1] - cuts[i - 1] \
                            + dp[i][partition - 1] \
                            + dp[partition + 1][j]
                    min_cost = min(cost, min_cost)
                
                dp[i][j] = min_cost

        return dp[1][c]





# Problem 43 - Burst Balloons
"""
* * Very Hard problem
https://leetcode.com/problems/burst-balloons/

You are given n balloons, indexed from 0 to n - 1. 
Each balloon is painted with a number on it represented by an array nums. 
You are asked to burst all the balloons.

If you burst the ith balloon, you will get 
nums[i - 1] * nums[i] * nums[i + 1] coins. 
If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a 
balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

Example 1:
Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167

video explanation : https://www.youtube.com/watch?v=Yz4LlDSlkns&list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY&index=52
"""
# Solution
"""
def f(i, j):
    if i > j:
        return 0

    min_cost = inf
    for partition = i -> j:
        cost = nums[i-1] * nums[partition] * nums[j + 1]
                + f(i, partition - 1)
                + f(partition + 1, j)
        
        min_cost = min(min_cost, cost)

    return min_cost

Note :  direction of i : 1 -> n
        direction of j : n -> 1

In tabulation i and j direction will be reveresed
"""
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)

        nums.insert(0, 1)
        nums.append(1)

        dp = [[0 for _ in range(n + 2)] for _ in range(n + 2)]

        for i in range(n, 0, -1):
            for j in range(1, n+1):
                if i > j: continue
                max_coins = -math.inf
                for partition in range(i, j+1):
                    coins = nums[i-1] * nums[partition] * nums[j + 1] + \
                           dp[i][partition - 1] + dp[partition +1][j]
                    max_coins = max(max_coins, coins)

                dp[i][j] = max_coins

        return dp[1][n]





# Problem 44 - Evaluate Boolean Expression to True
"""
Skipping this for now
"""




# Problem 45 - Palindrome Partitioning II
"""
https://leetcode.com/problems/palindrome-partitioning-ii/
Return the minimum cuts needed for a palindrome partitioning of s.

Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.
"""
# Solution
"""
pseudo code:
def f(i):
    if i == n:
        return 0
    min_cost = math.inf
    for j = i -> n:
        if isPalindrome(s[i : j]):   # j + 1 in python
            cost = 1 + f(i + 1)
            min_cost = min(cost, min_cost)
    return min_cost
"""





# Problem 46 - Partition Array for Maximum Sum
"""
https://leetcode.com/problems/partition-array-for-maximum-sum
Given an integer array arr, partition the array into (contiguous) 
subarrays of length at most k. 
After partitioning, each subarray has their values changed to become 
the maximum value of that subarray.
Return the largest sum of the given array after partitioning.

Example 1:
Input: arr = [1,15,7,9,2,5,10], k = 3
Output: 84
"""
# Solution : same as palindrome partitioning
"""
pseudo code :
k is at most k partition
def f(i):
    if i == n:
        return 0
    
    partition_len = 0
    max_sum = -inf
    max_elem = -inf

    for partition = i -> min(n-1, i + k):
        partition_len += 1
        max_elem = max(max_elem, arr[partition])

        sum = (partition_len * max_elem) + f(i + 1)

        max_sum = max(max_sum, sum)

    return max_sum
"""




# Problem 47 - Maximum Rectangle Area with all 1's
"""
https://leetcode.com/problems/maximal-rectangle/
Given a rows x cols binary matrix filled with 0's and 1's, 
find the largest rectangle containing only 1's and return its area.
"""
# Pre-requisite : Largest Rectangle in the histogram problem 
# Soultion : for every row if we can find the largest rectangle in the histogram
# and then among all the row return the max we have found
"""
pseudo code:

def f():
    heights = [0 for _ in range(cols)]
    max_area = 0

    for i = 0 -> row - 1:
        for j = 0 -> col -1:
            if matric[i][j] == 1:
                heights[j] += 1
            else:
                height[j] = 0

            area = largest_rectangle_in_histogram(heights)
            max_area = max(max_area, area)

    return max_area
""" 





# Problem 48 - Count Square Submatrices with All Ones
"""
https://leetcode.com/problems/count-square-submatrices-with-all-ones/
Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.
Example 1:
Input: matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There is  1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.
"""
# Solution : 
# Here in this problem the recursive function is not intuitive 
# we will directly solve by tabulation 

# https://www.youtube.com/watch?v=auS1fynpnjo

"""
pseudo code :
dp[0][j] = matrix[0][j]
dp[i][0] = matric[i][0]

for i=0 -> row - 1:
    for j=0 -> col - 1:
        dp[i][j] = min(dp[i-1][j], dp[i-1][-1], dp[i][j-1]) + 1

return sum(dp of all row and col values)
"""




# Problem 49 - Russian Doll Envelopes
"""
https://leetcode.com/problems/russian-doll-envelopes/
"""




# Problem 50 - Maximum Height by Stacking Cuboids 
"""
https://leetcode.com/problems/maximum-height-by-stacking-cuboids/
"""




# Problem 51 - Last Stone Weight II
"""
You are given an array of integers stones where stones[i] is the weight of the ith stone.
We are playing a game with the stones. On each turn, we choose any two stones and smash them 
together. Suppose the stones have weights x and y with x <= y. The result of this smash is:
* If x == y, both stones are destroyed, and
* If x != y, the stone of weight x is destroyed, and the stone of weight y has new weight y - x.
At the end of the game, there is at most one stone left.
Return the smallest possible weight of the left stone. If there are no stones left, return 0.

Input: stones = [2,7,4,1,8,1]
Output: 1
Explanation:
We can combine 2 and 4 to get 2, so the array converts to [2,7,1,8,1] then,
we can combine 7 and 8 to get 1, so the array converts to [2,1,1,1] then,
we can combine 2 and 1 to get 1, so the array converts to [1,1,1] then,
we can combine 1 and 1 to get 0, so the array converts to [1], then that's the optimal value.

Input: stones = [31,26,33,21,40]
Output: 5
"""
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # solved using memoization
        # concept of bounded knapsack

        # caching
        dp = {}

        stoneSum = sum(stones)
        target = math.ceil(stoneSum/2) 

        def f(idx, total):
            if total >= target or idx == len(stones):
                return abs(total - (stoneSum - total))

            if (idx, total) in dp:
                return dp[(idx, total)]

            # pick
            pick = f(idx + 1, total + stones[idx])

            # not pick
            not_pick = f(idx + 1, total)

            dp[(idx, total)] = min(pick, not_pick)

            return dp[(idx, total)]

        return f(0, 0)
    



# 52. Maximal Square
"""
https://leetcode.com/problems/maximal-square/description/
Given an m x n binary matrix filled with 0's and 1's, 
find the largest square containing only 1's and return its area.

Input: matrix = 
[["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
"""
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        ROWS, COLS = len(matrix), len(matrix[0])
        cache = {}  # map each (r, c) -> maxLength of square

        def helper(r, c):
            if r >= ROWS or c >= COLS:
                return 0

            if (r, c) not in cache:
                down = helper(r + 1, c)
                right = helper(r, c + 1)
                diag = helper(r + 1, c + 1)

                cache[(r, c)] = 0
                if matrix[r][c] == "1":
                    cache[(r, c)] = 1 + min(down, right, diag)
            return cache[(r, c)]

        helper(0, 0)
        return max(cache.values()) ** 2
    



# 53. Word Break
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




# 54. Longest Arithmetic Subsequence
"""
* * Good question
https://leetcode.com/problems/longest-arithmetic-subsequence
Given an array nums of integers, return the length of the longest arithmetic subsequence in nums.
A sequence seq is arithmetic if seq[i + 1] - seq[i] are all the same value (for 0 <= i < seq.length - 1).

Input: nums = [9,4,7,2,10]
Output: 3
Explanation:  The longest arithmetic subsequence is [4,7,10].

Input: nums = [20,1,15,3,10,5,8]
Output: 4
Explanation:  The longest arithmetic subsequence is [20,15,10,5].
"""
# Note read both examples carefully
# Question is similar to LIS
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        dp = {} # Key : <idx, difference>

        for idx in range(len(nums)):
            for prev in range(0, idx):
                dp[(idx, nums[idx] - nums[prev])] = \
                dp.get((prev, nums[idx] - nums[prev]), 1) + 1

        return max(dp.values())