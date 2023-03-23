# =======================================================================================================
# ==================================== RECURSIONS PROBLEMS ==============================================
# =======================================================================================================

# 1. Find all the subsequences of an array
"""
Note : here they have asked subsequence and not sub-array

       * * subesquence and Power set are same

Given a list arr of N integers, print all the subsequnce.
[1,2,3] ==> [],[1],[2],[3],[1,2],[2,3],[1,3],[1,2,3]

        [3, 1, 2]
        [p, X, X]   p - pick ; x - not pick
        [X, p, X]
        . . . . .
        . . . . .
        . . . . . 
"""
# Approach 1 : Recursion
# pick and non pick logic
def printSubesequences(nums: int) -> List[int]:
    res = []
    def recursion(idx, subSeq):
        if idx >= len(nums):
            res.append(subSeq)
            return
        
        # pick logic 
        subSeq.append(nums[idx])
        recursion(idx + 1, subSeq)
        subSeq.pop()

        # un-pick logic
        recursion(idx + 1, subSeq)

    recursion(0, [])
    return res
"""
Time complexity : 2^n
"""




# 1. Subset sum problem
"""
Given a list arr of N integers, print sums of all subsets.
"""
from typing import List
from unittest import result

def subsetSum(num: List[int]) -> List[int]:
    res = []
    # Each element decide to pick it or not pick it
    def recursion(ptr, sum):
        if ptr == len(num):
            res.append(sum)
            return

        # decision to pick current element
        recursion(ptr + 1, sum= sum + num[ptr])

        # decision to not pick the current element
        recursion(ptr + 1, sum)

    recursion(0, 0)
    return res
"""
Time complexity : 2^n
"""




# 2. Subsets
"""
Given an integer array nums of unique elements, return all possible subsets (the power set).
https://leetcode.com/problems/subsets/
"""
from typing import List

# Solution 1
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []
        def recursion(idx):
            if idx >= len(nums):
                res.append(subset.copy())
                return
            
            # include the element
            subset.append(nums[idx])
            recursion(idx + 1)

            # exclude the element
            subset.pop()
            recursion(idx + 1)

        recursion(0)
        return res



# =======================================================================================================
# ==================================== COMBINATION PROBLEMS =============================================
# =======================================================================================================
"""
        * * Note: combinations are subsequences only 
"""
# 3. Combination Sum
"""
https://leetcode.com/problems/combination-sum/
Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. 
You may return the combinations in any order.
The same number may be chosen from candidates an unlimited number of times. 
Two combinations are unique if the frequency of at least one of the chosen numbers is different.

Example 1 :
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]

Example 2 :
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def recursion(idx, combination, total):
            if total == target:
                res.append(combination.copy())
                return
            
            if idx >= len(candidates) or total > target:
                return

            # include the current element
            combination.append(candidates[idx])
            recursion(idx, combination, total + candidates[idx]) # same number can be choosen multiple times
            combination.pop()

            # exclude the current element
            recursion(idx + 1, combination, total)

        recursion(0, [], 0)
        return res

# Alternate solution with same logic
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def recursion(idx, combination, target):
            if idx == len(candidates):
                if target == 0:
                    res.append(combination.copy())
                return

            # include the current element
            if candidates[idx] <= target:
                combination.append(candidates[idx])
                # same number can be choosen multiple times
                recursion(idx, combination, target - candidates[idx])
                combination.pop()
            
            # exclude the current element
            recursion(idx + 1, combination, target)

        recursion(0, [], target)
        return res
"""
Time complexity : exponential (2^t * k)
"""




# 4. Combination Sum - II
"""
* * Good Question

https://leetcode.com/problems/combination-sum-ii/
Given a collection of candidate numbers (candidates) and a target number (target), 
find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.

Example 1:
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]

Example 2:
Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]

Note: The solution set must not contain duplicate combinations.
"""
# Approach 1 : just same as combination sum 
#              there are 2 change in recursion call
#              recursion(idx, combination, total + candidates[idx])
# 
# and use combination array as set to remove duplicates from the answer

# Approach 2  : as below
"""
    Important problem - difficult concept
    https://www.youtube.com/watch?v=G1fRTGRxXU8&list=PLgUwDviBIf0rGlzIn_7rsaR2FQ5e6ZOL9&index=10
"""
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort() 
        def recursion(idx, target, combination):
            if target == 0:
                res.append(combination.copy())
            if target < 0:
                return

            prev = -1
            for i in range(idx, len(candidates)):
                if candidates[i] == prev:
                    continue
                
                """
                here we are using backtracking approach
                and not pick & non_pick approach 
                """
                combination.append(candidates[i])
                recursion(i + 1, target - candidates[i], combination)
                # when recursion is over we chean-up for next backtracking
                combination.pop()

                prev = candidates[i]

        recursion(0, target, [])
        return res
# OR
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort() 
        def recursion(idx, target, combination):
            if target == 0:
                res.append(combination.copy())
                return
            if target < 0:
                return

            for i in range(idx, len(candidates)):
                if i > idx and candidates[i] == candidates[i-1]:
                    continue
                if candidates[i] > target:
                    break

                combination.append(candidates[i])
                recursion(i + 1, target - candidates[i], combination)
                combination.pop()

        recursion(0, target, [])
        return res




# 5. Palindrome Partitioning
"""
* * Good Question
Note : Palindrome is string which reads same from front and back
E.g. : aba or abba or aa etc

https://leetcode.com/problems/palindrome-partitioning/
Given a string s, partition s such that every substring of the partition is a palindrome. 
Return all possible palindrome partitioning of s.
A palindrome string is a string that reads the same backward as forward.
Example 1:
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]

Example 2:
Input: s = "a"
Output: [["a"]]
"""
# Approach :
# patition string at index i 
# and then look at the string from 0 to i-1 is it a valid palindrome 
# if yes  add it to result and continue with remaining right substring recursively
# if no   then move index + 1 and check is partiontioning causes the left substring to be palindrome 
"""https://www.youtube.com/watch?v=WBgsABoClE0&list=PLgUwDviBIf0rGlzIn_7rsaR2FQ5e6ZOL9&index=17"""
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        part = []

        def recursion(i):
            if i == len(s):
                res.append(part.copy())
                return
            
            for j in range(i, len(s)):
                if self.is_palindrome(s, i, j):
                    part.append(s[i: j+1])
                    recursion(j+1)
                    part.pop()
            
        recursion(0)
        return res

    def is_palindrome(self, s, i, j):
        while i < j:
            if s[i] != s[j]:
                return False
            i, j = i + 1, j - 1
        return True



# =======================================================================================================
# ==================================== PERMUTATION PROBLEMS =============================================
# =======================================================================================================

# 6. Permutations
"""
https://leetcode.com/problems/permutations/
Given an array nums of distinct integers, return all the possible permutations. 
You can return the answer in any order.
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
"""
"""https://www.youtube.com/watch?v=f2ic2Rsc9pU&list=PLgUwDviBIf0rGlzIn_7rsaR2FQ5e6ZOL9&index=13"""
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(index):
            if index == len(nums):
                res.append(nums[:])
                return

            for i in range(index, len(nums)):
                # swap
                nums[index], nums[i] = nums[i], nums[index]
                # backtrack
                backtrack(index + 1)
                # undo changes
                nums[index], nums[i] = nums[i], nums[index]

        backtrack(0)
        return res




# 7. Next Permutation
"""
https://leetcode.com/problems/next-permutation/

For example, the next permutation of arr = [1,2,3] is [1,3,2].
Similarly, the next permutation of arr = [2,3,1] is [3,1,2].

https://youtu.be/LuLCLgMElus?t=137
"""
"""
    Pseudo code:
        1. Find the dip point 
        Linearly traverse nums from backwards 
           and find first index `i` such that nums[i] < nums[i+1] --> say this as index-1
           For example:
              1  3    5    4   2
                 ^    ^
                 i   i+1

            here index-1 is value 3 (having index 1)
            i.e find the dip point from backward (as u can see no.s started to increase 
            from backwards and at 3 there is a dip)

        2. Again linearly traverse nums from backwards 
           and find first index (say index-2) which is greater than value at index-1
           In the example above its 4 (having index 3)
        3. swap(index-1, index-2)
        4. Reverse everything from (index-1 + 1 to n)
"""




# 9. N-Queen Problem
"""
https://leetcode.com/problems/n-queens/

Hint : (r+c) for positive diagonal
       (r-c) for negative diagonal
"""
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        board = [["."] * n for i in range(n)]

        res = []
        col = set()
        pos_diag = set()
        neg_diag = set()
        def backtrack(r):
            if r == n:
                res.append(["".join(row) for row in board])
                return
            
            for c in range(n):
                if c in col or (r+c) in pos_diag or (r-c) in neg_diag:
                    continue

                col.add(c)
                pos_diag.add(r+c)
                neg_diag.add(r-c)
                board[r][c] = "Q"

                backtrack(r + 1)
                
                col.remove(c)
                pos_diag.remove(r+c)
                neg_diag.remove(r-c)
                board[r][c] = "."

        backtrack(0)
        return res




# 10. M-Coloring Problem
"""
https://leetcode.com/problems/flower-planting-with-no-adjacent/description/
Given an undirected graph and an integer M. 
The task is to determine if the graph can be colored with at most M colors 
such that no two adjacent vertices of the graph are colored with the same color.
"""
class Solution:
    def graphColoring(self, m:int, graph:dict) -> bool:
        n = len(graph)
        node_color = {i:0 for i in range(n)}

        def color_posible(node, color):
            # check if adjacent node is of same color as that of the node
            for neighbour in graph[node]:
                if node_color[neighbour] == color:
                    return False
            return True

        def backtrack(node, color):
            # when no node is remaining and all nodes have been colored -> return True
            if node == n:
                return True

            for _color in range(1, m+1):
                # is it possible to color a node with _color
                if color_posible(node, _color):
                    node_color[node] = color
                    if backtrack(node + 1) == True:
                        return True
                    node_color[node] = 0
            
            return False

        if backtrack(0, m) == True:
            return True

        return False




# 8. Generate Parentheses
"""
https://leetcode.com/problems/generate-parentheses/
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
"""
# Solution : 
# for 3 pair of parentheses we have 3 opening parentheses and 3 closing parentheses 
# we can solve the problem using - backtracking considering the equation that
# we can add opening parentheses if open_cnt < n and 
# we can only add a closing parentheses to our result if open_cnt > close_cnt
# we will solve this problem using backtracking 


class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        # we have two choices to make either we can add open parentheses or a closing parentheses
        def backtrack(parentheses_str, open_cnt, close_cnt):
            if open_cnt == close_cnt == n:
                result.append(parentheses_str)
                return

            # add open parenthesis
            if open_cnt < n:
                parentheses_str = parentheses_str + "("
                backtrack(parentheses_str, open_cnt + 1, close_cnt)
                parentheses_str = parentheses_str[:-1]

            # add the closing parenthesis
            if close_cnt < open_cnt:
                parentheses_str = parentheses_str + ")"
                backtrack(parentheses_str, open_cnt, close_cnt + 1)
                parentheses_str = parentheses_str[:-1]

        backtrack("", 0, 0)
        return result




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# OPTIONAL 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 9. K-th Permutation Sequence ==> HARD
"""
* * VERY HARD
Given n and k, return the kth permutation sequence.
https://leetcode.com/problems/permutation-sequence/
"""
## Basic approach --> recursively find all permutation, 
#                     sort the permutation result, 
#                     return the kth permutation

# Best Approach : 
# https://www.youtube.com/watch?v=wT7gcXLYoao&list=PLgUwDviBIf0rGlzIn_7rsaR2FQ5e6ZOL9&index=19
import math
class Solution:
    def getPermutation(self, n: int, k: int) -> str:

        nums = [i for i in range(1, n+1)]

        # 1. find factorial of k-1
        new_n = n - 1
        fact_new_n = math.factorial(new_n)

        # 2. since zero base indexing reduce k by 1
        k = k -1

        # ans string
        result = ""

        while True:
            result = result + str(nums[int(k/fact_new_n)])
            nums.remove(nums[int(k/fact_new_n)])
            if len(nums) == 0:
                break

            k = k % fact_new_n
            fact_new_n = fact_new_n / len(nums)

        return result
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# 10. Beautiful Arrangement
"""
https://leetcode.com/problems/beautiful-arrangement/
Suppose you have n integers labeled 1 through n. A permutation of those n integers perm (1-indexed) 
is considered a beautiful arrangement if for every i (1 <= i <= n), either of the following is true:
1. perm[i] is divisible by i.
2. i is divisible by perm[i].
Given an integer n, return the number of the beautiful arrangements that you can construct.

Input: n = 2
Output: 2
Explanation: 
The first beautiful arrangement is [1,2]:
    - perm[1] = 1 is divisible by i = 1
    - perm[2] = 2 is divisible by i = 2
The second beautiful arrangement is [2,1]:
    - perm[1] = 2 is divisible by i = 1
    - i = 2 is divisible by perm[2] = 1
"""
# Solution : After swapping in backtracking check if the generated permutaion is valid
# rest code is all same
class Solution:
    def countArrangement(self, n: int) -> int:
        count = 0
        arr = list(range(1, n+1))

        def backtracking(index):
            nonlocal count
            if index == n:
                count += 1
                return

            for i in range(index, n):
                arr[index], arr[i] = arr[i], arr[index]
                # check if the current is valid arrangement
                if arr[index] % (index+1) == 0 or (index+1) % arr[index] == 0:
                    backtracking(index+1)
                arr[index], arr[i] = arr[i], arr[index]

        backtracking(0)
        return count