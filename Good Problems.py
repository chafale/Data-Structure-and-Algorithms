from ast import List

# 1. Merge Sort
"""
Divide and conquer technique

pseudo code:
    global array a[low : high]

    def MergeSort(low, high):
        if low < high:
            mid = (low + high)//2

            # divide step
            MergeSort(low, mid)
            MergeSort(mid+1, high)

            # conquer or combine step
            Merge(low, mid, high)

    def Merge(low, mid, high):
        # b -> temporary array just to store results
        # a -> is the global array a[low : mid] and a[mid + 1 : high]
        # k is pointer to b array
        # i is pointer to a[low : mid]
        # j is pointer to a[mid + 1 : high]

        k = low, i = low, j = mid

        while i <= mid and j <= high:
            if a[i] < a[j]:
                b[k] = a[i]
                i += 1
            else:
                b[k] = a[j]
                j += 1

            k += 1

        # add remaing items from a[low : mid] or a[mid + 1 : high] to b array

        if i > mid:
            while j <= high:
                b[k] = a[j]
                j += 1
                k += 1
        else:
            while i <= mid:
                b[k] = a[i]
                i += 1
                k += 1

        # copy b array to a array
        a[low : high] = b.copy()

"""

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



# 3. Next Permutation
"""
https://leetcode.com/problems/next-permutation/
Example :
        Input: nums = [1,2,3]
        Output: [1,3,2]
"""
"""
    Pseudo code:
        1. Linearly traverse nums from backwards 
           and find first index `i` such that nums[i] < nums[i+1] --> say this as index-1
        2. Again linearly traverse nums from backwards 
           and find first index (say index-2) which is greater than value at index-1
        3. swap(index-1, index-2)
        4. Reverse everything from (index-1 + 1 to n)
"""




# 4. My Calendar I
"""
https://leetcode.com/problems/my-calendar-i/
Implement the MyCalendar class:
1. MyCalendar() Initializes the calendar object.
2. boolean book(int start, int end) Returns true if the event can be added to the calendar 
    successfully without causing a double booking. 
    Otherwise, return false and do not add the event to the calendar.
"""
class MyCalendar:

    def __init__(self):
        self.events = []
        

    def book(self, start: int, end: int) -> bool:
        for s, e in self.events:
            # cool condition to check if intervals are overlapping
            if s < end and start < e:
                return False
        self.events.append((start, end))
        return True
    



# 5. Inversion count in Array using Merge Sort
"""
Inversion Count for an array indicates – how far (or close) the array is from being sorted. 
If the array is already sorted, then the inversion count is 0, but if the array is sorted 
in reverse order, the inversion count is the maximum. 

Given an array a[]. The task is to find the inversion count of a[]. 
Where two elements a[i] and a[j] form an inversion if a[i] > a[j] and i < j.
"""
# Hint 1: 
# so we have to find all pairs such that a[i] > a[j 
# 
# Hint 2: use merge sort logic 




