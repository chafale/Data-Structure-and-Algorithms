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
https://leetcode.com/problems/largest-rectangle-in-histogram/
Given an array of integers heights representing the histogram's bar height 
where the width of each bar is 1, return the area of the largest rectangle in the histogram.
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
# we will seaprate each number in to bucket of their count : 
# and bucket will contain list of elements
# we will pop element from the max count bucket
class FreqStack:

    def __init__(self):
        self.frequency_map = {}  # we will have frequency count of elements
        self.maxCount = 0
        self.freq_cnt_grp = defaultdict(list)        

    def push(self, val: int) -> None:
        val_freq = 1 + self.frequency_map.get(val, 0)
        self.frequency_map[val] = val_freq
        if val_freq > self.maxCount:
            self.maxCount = val_freq
        self.freq_cnt_grp[val_freq].append(val)
        
    def pop(self) -> int:
        pop_elem = self.freq_cnt_grp[self.maxCount].pop()
        self.frequency_map[pop_elem] -= 1
        if not self.freq_cnt_grp[self.maxCount]:
            self.maxCount -= 1
        return pop_elem
  



# 4. 132 Pattern
"""
* deceptive medium hard
https://leetcode.com/problems/132-pattern/

Given an array of n integers nums, a 132 pattern is a subsequence of three integers 
nums[i], nums[j] and nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].

Return true if there is a 132 pattern in nums, otherwise, return false.

Input: nums = [3,1,4,2]
Output: true
Explanation: There is a 132 pattern in the sequence: [1, 4, 2].

Input: nums = [1,2,3,4]
Output: false
Explanation: There is no 132 pattern in the sequence.

Input: nums = [-1,3,2,0]
Output: true
Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].
"""
# monotonically decreasing stack
# video link : https://www.youtube.com/watch?v=q5ANAl8Z458
class Solution: 
    def find132pattern (self, nums : List[int]) -> bool: 
        stack = [] # pair [num, minLeft], mono decreasing 
        curMin = nums[0] 
        for n in nums [1:]: 
            # here n is k
            while stack and n >= stack[-1][0]: 
                stack.pop() 

            if stack and n < stack[-1][0] and n > stack[-1][1]: 
                return True 
            
            stack.append ([n, curMin]) 
            curMin = min (curMin, n) 
        return False
    



# 5. Remove All Adjacent Duplicates in String II
"""
https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
We repeatedly make k duplicate removals on s until we no longer can.

Input: s = "deeedbbcccbdaa", k = 3
Output: "aa"
Explanation: 
First delete "eee" and "ccc", get "ddbbbdaa"
Then delete "bbb", get "dddaa"
Finally delete "ddd", get "aa"
"""
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        stack = []  # [char, count]

        for c in s:
            if stack and stack[-1][0] == c:
                stack[-1][1] += 1
            else:
                stack.append([c, 1])

            if stack[-1][1] == k:
                stack.pop()

        res = ""
        for char, count in stack:
            res += char * count

        return res




# 6. Remove K Digits
"""
* Deceptive problem
https://leetcode.com/problems/remove-k-digits
Given string num representing a non-negative integer num, and an integer k, 
return the smallest possible integer after removing k digits from num.

Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 
1219 which is the smallest.

Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not 
contain leading zeroes.
"""
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        numStack = []
        
        # Construct a monotone increasing sequence of digits
        for digit in num:
            while k and numStack and numStack[-1] > digit:
                numStack.pop()
                k -= 1
        
            numStack.append(digit)
        
        # - Trunk the remaining K digits at the end
        # - in the case k==0: return the entire list
        finalStack = numStack[:-k] if k else numStack
        
        # trip the leading zeros
        return "".join(finalStack).lstrip('0') or "0"
    



# 7. Decode String
"""
https://leetcode.com/problems/decode-string/
Given an encoded string, return its decoded string.
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Input: s = "3[a2[c]]"
Output: "accaccacc"
"""
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []

        for char in s:
            if char is not "]":
                stack.append(char)
            else:
                sub_str = ""
                while stack[-1] is not "[":
                    sub_str = stack.pop() + sub_str
                stack.pop()

                multiplier = ""
                while stack and stack[-1].isdigit():
                    multiplier = stack.pop() + multiplier

                stack.append(int(multiplier) * sub_str)

        return "".join(stack)




# 8. Simplify Path
"""
Given a string path, which is an absolute path (starting with a slash '/') to a file or 
directory in a Unix-style file system, convert it to the simplified canonical path.

The canonical path should have the following format:

The path starts with a single slash '/'.
Any two directories are separated by a single slash '/'.
The path does not end with a trailing '/'.
The path only contains the directories on the path from the root directory to the target file 
or directory (i.e., no period '.' or double period '..')

Return the simplified canonical path.
"""
class Solution:
    def simplifyPath(self, path: str) -> str:

        stack = []

        for i in path.split("/"):
            #  if i == "/" or i == '//', it becomes '' (empty string)

            if i == "..":
                if stack:
                    stack.pop()
            elif i == "." or i == '':
                # skip "." or an empty string
                continue
            else:
                stack.append(i)

        res = "/" + "/".join(stack)
        return res
    



# 9. Car Fleet
"""
https://leetcode.com/problems/car-fleet/
There are n cars going to the same destination along a one-lane road. 
The destination is target miles away.

You are given two integer array position and speed, both of length n, 
where position[i] is the position of the ith car and speed[i] is the speed 
of the ith car (in miles per hour).

A car can never pass another car ahead of it, but it can catch up to it and drive bumper 
to bumper at the same speed. The faster car will slow down to match the slower car's speed. 
The distance between these two cars is ignored (i.e., they are assumed to have the same position).

A car fleet is some non-empty set of cars driving at the same position and same speed. 
Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered 
as one car fleet.

Return the number of car fleets that will arrive at the destination.

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
Explanation:
The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12.
The car starting at 0 does not catch up to any other car, so it is a fleet by itself.
The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. 
The fleet moves at speed 1 until it reaches target.
Note that no other cars meet these fleets before the destination, so the answer is 3.
"""
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        pair = [(p, s) for p, s in zip(position, speed)]
        pair.sort(reverse=True)
        stack = []
        for p, s in pair:  # Reverse Sorted Order
            stack.append((target - p) / s)
            if len(stack) >= 2 and stack[-1] <= stack[-2]:
                stack.pop()
        return len(stack)




# 10. Online Stock Span
"""
https://leetcode.com/problems/online-stock-span/
Design an algorithm that collects daily price quotes for some stock and 
returns the span of that stock's price for the current day.

The span of the stock's price in one day is the maximum number of consecutive days 
(starting from that day and going backward) for which the stock price was less than or 
equal to the price of that day.

Implement the StockSpanner class:
StockSpanner() Initializes the object of the class.
int next(int price) Returns the span of the stock's price given that today's price is price.
"""
# Monontonic decreasing stack
class StockSpanner:
    def __init__(self):
        self.stack = []  # pair: (price, span)

    def next(self, price: int) -> int:
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack[-1][1]
            self.stack.pop()
        self.stack.append((price, span))
        return span




# 11. Daily Temperatures
"""
https://leetcode.com/problems/daily-temperatures/
Given an array of integers temperatures represents the daily temperatures, 
return an array answer such that answer[i] is the number of days you have to wait 
after the ith day to get a warmer temperature. 
If there is no future day for which this is possible, keep answer[i] == 0 instead.

Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
"""
# Monotonic decreasing stack
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []  # pair: [temp, index]

        for i, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = i - stackInd
            stack.append((t, i))
        return res




# 12. Asteroid Collision
"""
https://leetcode.com/problems/asteroid-collision/
We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its 
direction (positive meaning right, negative meaning left). Each asteroid moves at 
the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller 
one will explode. If both are the same size, both will explode. Two asteroids moving in the 
same direction will never meet.

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.
"""
class Solution(object):
    def asteroidCollision(self, asteroids):
        ans = []
        for new in asteroids:
            while ans and new < 0 < ans[-1]:
                if ans[-1] < -new:
                    ans.pop()
                    continue
                elif ans[-1] == -new:
                    ans.pop()
                break
            else:
                ans.append(new)
        return ans
    



# 13. Generate Parentheses
"""
https://leetcode.com/problems/generate-parentheses/
Given n pairs of parentheses, write a function to generate all combinations of 
well-formed parentheses.
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
"""
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
    



# 14. Evaluate Reverse Polish Notation
"""
https://leetcode.com/problems/evaluate-reverse-polish-notation/

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
"""
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                a, b = stack.pop(), stack.pop()
                stack.append(b - a)
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                a, b = stack.pop(), stack.pop()
                stack.append(int(b / a))
            else:
                stack.append(int(c))
        return stack[0]




# 15. Min Stack
"""
https://leetcode.com/problems/min-stack/
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.
"""
# Hint : Consider each node in the stack having a minimum value. 
class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]




# 16. Basic Calculator II
"""
https://leetcode.com/problems/basic-calculator-ii
Given a string s which represents an expression, evaluate this expression and return its value. 

Input: s = "3+2*2"
Output: 7

Input: s = " 3/2 "
Output: 1
"""
class Solution:
    def calculate(self, s: str) -> int:
        operator = {"+", "-", "*", "/"}

        stack = []
        prevOperator = "+"

        i = 0
        currNumber = 0
        while i < len(s):
            if s[i].isdigit():
                currNumber = currNumber * 10 + int(s[i])

            if s[i] in operator or i == len(s) - 1:
                if prevOperator == "+":
                    stack.append(currNumber)
                elif prevOperator == "-":
                    stack.append(-1 * currNumber)
                elif prevOperator == "*":
                    stack.append(stack.pop() * currNumber)
                else:
                    stack.append(int(stack.pop()/currNumber))
                currNumber = 0
                prevOperator = s[i]
            i += 1

        return sum(stack)