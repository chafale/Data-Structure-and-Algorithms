from ast import List
from collections import defaultdict
from collections import deque 
from typing import Optional
import heapq
import math

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

head, prev = None, None

# =======================================================================================================
# ==================================== LINKED LIST PROBLEMS =============================================
# =======================================================================================================

# Basic 1 - Reversing the LL
"""
https://leetcode.com/problems/reverse-linked-list/
Given the head of a singly linked list, reverse the list, and return the reversed list.
"""
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        # Approach 1 : using Iteration
        def reverseLL(head):
            prev = None
            curr = head
            nxt = curr.next

            while curr.next:
                curr.next = prev
                
                prev = curr
                curr = nxt
                nxt = nxt.next

            curr.next = prev
            return curr

        return reverseLL(head)

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        # Approach 2 : using recursion
        def dfs(node, parent):
            nonlocal head
            if node.next is None:
                node.next = parent
                head = node
                return

            dfs(node.next, node)

            node.next = parent

            return node

        dfs(head, None)
        return head

    

# Basic 2 - Middle of the Linked List
"""
https://leetcode.com/problems/middle-of-the-linked-list/
Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

Input: head = [1,2,3,4,5]
Output: 3
Explanation: The middle node of the list is node 3.

Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
"""
# Sol : Using fast and slow pointer
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow




# Basic 3 - Inplace merge two Linklist
# merge two sorted linked lists [Problem 2]
# merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
head1 = ListNode("1->2->3->4")
head2 = ListNode("6->5->4")
# here second will always points to shorter list
first, second = head1, head2
while second.next:
    tmp = first.next
    first.next = second
    first = tmp
    
    tmp = second.next
    second.next = first
    second = tmp




# Problem 1 - Reorder List
"""
https://leetcode.com/problems/reorder-list/
You are given the head of a singly linked-list. The list can be represented as:
L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
Do not return anything, modify head in-place instead.

Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
"""
# Solution : we want to alternate the increasing and decreasing values
# 1. Find the mid point of the list using fast and slow pointer
# 2. Reverse the second half of the list
# 3. Merge the first half and the second half




# Problem 2 - Merge Two Sorted Lists
"""
https://leetcode.com/problems/merge-two-sorted-lists
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. 
The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
"""
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        head = ListNode(-1)

        tail = head
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next            
            tail = tail.next

        tail.next = l1 if l1 is not None else l2

        return head.next




# Problem 3 - Remove Nth Node From End of List
"""
https://leetcode.com/problems/remove-nth-node-from-end-of-list/
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
"""
# Solution : 
# we could use two pointers. 
# The first pointer advances the list by n+1 steps from the beginning,
# while the second pointer starts from the beginning of the list. 
# 
# Now, both pointers are exactly separated by `n` nodes apart. 
# We maintain this constant gap by advancing both pointers together 
# until the first pointer arrives past the last node. 
# The second pointer will be pointing at the n-th node counting from the last. 
# 
# We relink the next pointer of the node referenced by the second pointer to point 
# to the node's next next node. 

"""https://www.youtube.com/watch?v=XVuQxVej6y8&list=PLot-Xpze53leU0Ec0VkBhnf4npMRFiNcB&index=8"""
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)

        """
        dummy 
        |
        `---> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
                              nth

        slow = dummy 
        and fast = k nodes (apart) from slow
        """
        left = dummy
        right = head

        # adjust right k nodes apart from left
        while n > 0:
            right = right.next
            n -= 1

        while right:
            left = left.next
            right = right.next

        # delete
        left.next = left.next.next
        return dummy.next





# Problem 4 - Detect a cycle in LL (OR) Floyd's Tortoise and Hare Algorithm
"""
https://leetcode.com/problems/linked-list-cycle/
Given head, the head of a linked list, determine if the linked list has a cycle in it.
"""
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
            
        return False




# Problem 5 - Linked List Cycle II
"""
https://leetcode.com/problems/linked-list-cycle-ii/
Given the head of a linked list, return the node where the cycle begins. 
If there is no cycle, return null.
"""
# Solution : 
# 1. using floyd's algo find the collision point (where slow and fast ptr meet)
#    using slow and fast pointer
# 2. use an another pointer pointing to head -> say tmp
#    move slow and tmp by one steps symultaneously. The node when both of them meet is 
#    the start point




# Problem 6 - Find the Duplicate Number
"""
* * Improtant problem -- considered to be hard
https://leetcode.com/problems/find-the-duplicate-number/
Given an array of integers nums containing n + 1 integers where each integer 
is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

Example 1:
Input: nums = [1,3,4,2,2]
Output: 2

Example 2:
Input: nums = [3,1,3,4,2]
Output: 3

https://www.youtube.com/watch?v=wjYnzkAhcNk
"""
# Solution : think nums not as values but as pointers to index of the array.
# so it will create a graph having cycle
# Therefore first we will apply floyd's algo (slow-fast pointer algo) to find 
# start point of the cycle of LL and return the start point
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        tmp = 0
        while True:
            tmp = nums[tmp]
            slow = nums[slow]
            if tmp == slow:
                return slow




# Problem 7 - Remove Duplicates from Sorted List
"""
https://leetcode.com/problems/remove-duplicates-from-sorted-list/
Given the head of a sorted linked list, delete all duplicates such that each element 
appears only once. Return the linked list sorted as well.
"""       
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        while curr:
            while curr.next and curr.val == curr.next.val:
                curr.next = curr.next.next

            curr = curr.next

        return head        




# Problem 8 - Intersection of Two Linked Lists
"""
https://leetcode.com/problems/intersection-of-two-linked-lists/
Given the heads of two singly linked-lists headA and headB, 
return the node at which the two lists intersect. 
"""
# Solution : We use two pointers p1 and p2 to start from two LL A and B 
# we will interate ptr as p1 = p1.next, p2 = p2.next if any of the pointer reaches
# null we will assign the pointer to the opposite LL
# same we will do for other pointer which will eventually reach null after some time 
# and assign it to opposite LL
# Now, this time when two pointers meet that is the intersection point
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pa, pb = headA, headB

        while pa != pb:
            pa = pa.next if pa else headB
            pb = pb.next if pb else headA

        return pa       
    


# Problem 9 - Flatten a Multilevel Doubly Linked List
"""
https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
You are given a doubly linked list, which contains nodes that have a next pointer, a previous pointer, 
and an additional child pointer. This child pointer may or may not point to a separate doubly linked list,
 also containing these special nodes. These child lists may have one or more children of their own, 
 and so on, to produce a multilevel data structure as shown in the example below.

Given the head of the first level of the list, flatten the list so that all the nodes appear in a 
single-level, doubly linked list. Let curr be a node with a child list. The nodes in the child list 
should appear after curr and before curr.next in the flattened list.

Return the head of the flattened list. The nodes in the list must have all of their child pointers 
set to null.

Input: head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
Output: [1,2,3,7,8,11,12,9,10,4,5,6]
"""
class Solution:
    def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
        last = None
        def flatten_LL(node):
            nonlocal last
            if not node:
                return None

            flatten_LL(node.next)
            flatten_LL(node.child)

            node.next = last
            node.child = None
            if last:
                last.prev = node
            last = node

        flatten_LL(head)
        return head
    



# 10 . Copy List with Random Pointer
"""
https://leetcode.com/problems/copy-list-with-random-pointer
Construct a deep copy of the list. 
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
"""
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        old_new = {}
        def dfs(node):
            if not node:
                return None

            if node in old_new:
                return old_new[node]

            new_node = Node(node.val)
            old_new[node] = new_node

            new_node.next = dfs(node.next)
            new_node.random = dfs(node.random)

            return new_node
        return dfs(head)
    



# 11. Reverse Nodes in k-Group
"""
* * Good Problem -- Hard difficulty
https://leetcode.com/problems/reverse-nodes-in-k-group
Given the head of a linked list, reverse the nodes of the list k at a time, 
and return the modified list.

Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        def findKth(curr, k):
            while curr and k > 0:
                curr = curr.next
                k -= 1
            return curr

        dummy = ListNode(-1, head)
        grpPrev = dummy

        while True:
            kth = findKth(grpPrev, k)
            if not kth:
                break

            grpNext = kth.next

            # reverse the linkedList
            prev, curr = grpNext, grpPrev.next
            while curr != grpNext:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp

            # * * this portion is tricky
            tmp = grpPrev.next
            grpPrev.next = kth
            grpPrev = tmp

        return dummy.next
    



# 12. Reverse the linklist - ii
"""
https://leetcode.com/problems/reverse-linked-list-ii/description/
Given the head of a singly linked list and two integers left and right where left <= right, 
reverse the nodes of the list from position left to position right, and return the reversed list.
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
"""
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)

        # 1) reach node at position "left"
        leftPrev, cur = dummy, head
        for i in range(left - 1):
            leftPrev, cur = cur, cur.next

        # Now cur="left", leftPrev="node before left"
        # 2) reverse from left to right
        prev = None
        for i in range(right - left + 1):
            tmpNext = cur.next
            cur.next = prev
            prev, cur = cur, tmpNext

        # 3) Update pointers
        leftPrev.next.next = cur  # cur is node after "right"
        leftPrev.next = prev  # prev is "right"
        return dummy.next




# 13. Merge k Sorted Lists
"""
https://leetcode.com/problems/merge-k-sorted-lists
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
"""
from queue import PriorityQueue

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        # this is implement becoz initial Priority queue does not support comparison between 
        # objects
        class Wrapper():
            def __init__(self, node) -> None:
                self.node = node
            def __lt__(self, other):
                # this function is used by PriorityQueue for comparison
                return self.node.val < other.node.val

        # new list
        head = tail = ListNode(0)

        q = PriorityQueue()

        for l in lists:
            if l:
                q.put(Wrapper(l)) # put intial node

        while not q.empty():
            node = q.get().node # get() function will return Wrapper object we need node object
            tail.next = node
            tail = tail.next
            node = node.next
            if node:
                q.put(Wrapper(node))
        return head.next