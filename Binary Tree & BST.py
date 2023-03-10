from ast import List
import sys
import math
from typing import Optional
import collections
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# =======================================================================================================
# ==================================== BINARY TREE PROBLEMS =============================================
# =======================================================================================================

# Basic 1 : Traversal technique 
# Tree Traversal techniques : 
# 1. BFS : Level order traversal
# 2. DFS :  IN-ORDER, PRE-ORDER, POST-ORDER

# These are all DFS traversal
def preOrder(node):
    if not node:
        return
    print(node.val)
    preOrder(node.left)
    preOrder(node.right)

def inOrder(node):
    if not node:
        return
    inOrder(node.left)    
    print(node.val)
    inOrder(node.right)

def postOrder(node):
    if not node:
        return
    postOrder(node.left)
    postOrder(node.right)
    print(node.val)

""" 
Time complexity : O(n)
Space complexity : O(n)
"""

def bfs(node):
    # also called as level order traversal
    queue = []
    queue.append(node)

    while queue:
        pop_elem = queue.pop(0)
        # process the pop element

        # add pop_elem children in the queue
        if pop_elem.left:
            queue.append(pop_elem.left)
        if pop_elem.right:
            queue.append(pop_elem.right)
""" 
Time complexity : O(n)
Space complexity : O(n)
"""



# Basic 2 : Height of Binary Tree (or) Depth of the Binary Tree
def heightBT(node):
    if not node:
        return 0
    
    lh = heightBT(node.left)      #left height
    rh = heightBT(node.right)     #right height

    return 1 + max(lh, rh)
"""
Time Complexity : O(n)
"""


# Basic 3 : Print all leaves of BT
def leafBoundry(node):
    if not node.left and not node.right:
        print(node.val)

    if node.left:
        leafBoundry(node.left)
    if node.right:
        leafBoundry(node.right)



# Basic 4 : check whether two trees are same
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p == q

        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False



# Basic 5 : Morris Traversal | InOrder
"""
The resursive traversal takes O(n) - TC and O(n) - SC
Morris traversal takes O(n) - TC but takes O(1) - SC - does not consumes any space

Morris traversal uses the concept of threaded binary tree
"""





# Problem 1 - Check for Balanced Binary Tree
"""
https://leetcode.com/problems/balanced-binary-tree/
Given a binary tree, determine if it is 
height-balanced.

* * A height-balanced binary tree is a binary tree in which the 
depth of the two subtrees of every node never differs by more than one.
"""
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def heightBT(node):
            if not node:
                return 0, True
            
            lh, l_isBalanced = heightBT(node.left)
            rh, r_isBalanced = heightBT(node.right)

            node_isBalanced = False
            if l_isBalanced and r_isBalanced and abs(lh - rh) <= 1:
                node_isBalanced = True

            return 1 + max(lh, rh), node_isBalanced

        return heightBT(root)[1]




# Problem 2 - Diameter of Binary Tree
"""
https://leetcode.com/problems/diameter-of-binary-tree/
Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes 
in a tree. This path may or may not pass through the root.
"""
# diameter at a node = left_height + right_height
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_diameter = 0
        def heightBT(node):
            nonlocal max_diameter

            if not node:
                return 0

            lh, rh = heightBT(node.left), heightBT(node.right)

            diameter = lh + rh
            max_diameter = max(max_diameter, diameter)

            return 1 + max(lh, rh)

        heightBT(root)
        return max_diameter




# Problem 3 - Binary Tree Maximum Path Sum
"""
https://leetcode.com/problems/binary-tree-maximum-path-sum/
Given the root of a binary tree, return the maximum path sum of any non-empty path.
"""
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        max_path_sum = -math.inf
        def postOrder(node):
            nonlocal max_path_sum

            if not node:
                return 0

            """
            * * note : here do not consider negative values as they will not be in max path sum
            so we will have max(0,postOrder(node.left/right))
            """
            lf_val = max(0, postOrder(node.left))
            rt_val = max(0, postOrder(node.right))

            path_sum = node.val + lf_val + rt_val
            max_path_sum = max(path_sum, max_path_sum)

            return node.val + max(lf_val, rt_val)

        postOrder(root)
        return max_path_sum




# Problem 4 - Check it two trees are Identical or Not
"""
https://leetcode.com/problems/same-tree/
Given the roots of two binary trees p and q, write a function to check if they are the same or not.
"""
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True

        if not p and q:
            return False
        
        if p and not q:
            return False

        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False

# (or a concise solution)
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p == q

        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False




# Problem 5 - Zigzag Level Order Traversal
"""
https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal
"""
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        q = collections.deque()
        q.append(root)
        res = []
        _dir = 1
        while q:
            level = []
            q_len = len(q)
            for i in range(q_len):
                tmp = q.popleft()
                level.append(tmp.val)
                if tmp.left:
                    q.append(tmp.left)
                if tmp.right:
                    q.append(tmp.right)
            res.append(level[::_dir])
            _dir *= -1
            
        return res  




# Problem 6 - Boundary Traversal in Binary Tree
"""
https://leetcode.com/problems/boundary-of-binary-tree/
"""
# Solution : for anti-clockwise movement
#  1. Find the left boundary (except the leaf nodes)
#  2. Find the leaf nodes
#  3. Find the right boundry (except the leaf nodes) <then reverse the list> 
"""
pseudo code:
res = []

def leftBoundry(node):
    # using this else-if we are eliminating leaf nodes
    if node.left:
        res.append(node.val)
        leftBoundry(node.left)
    elif node.right:
        res.append(node.val)
        leftBoundry(node.right)

def leafBoundry(node):
    if not node.left and not node.right:
        res.append(node.val)

    if node.left:
        leafBoundry(node.left)
    if node.right:
        leafBoundry(node.right)

# Note: while printing the right-boundry we will not call the function from the root
# instead we will call from root.right as we don't want to print root twice
right_boundry = []
def rightBoundry(node):
    if node.right:
        right_boundry.append(node.val)
        leftBoundry(node.right)
    elif node.left:
        right_boundry.append(node.val)
        leftBoundry(node.left) 
# we will reverse the right_boundry list and append to the result

res = res + right_boundry[::-1]

return res
"""




# Problem 7 - Vertical Order Traversal of a Binary Tree
"""
https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/
Input: root = [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
"""
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        level_order = collections.defaultdict(list)
        min_v_order = 0
        max_v_order = 0

        def dfs(node, v_level):
            nonlocal min_v_order, max_v_order
            
            if not node:
                return

            level_order[v_level].append(node.val)
            min_v_order = min(min_v_order, v_level)
            max_v_order = max(max_v_order, v_level)

            dfs(node.left, v_level - 1)
            dfs(node.right, v_level + 1)

        dfs(root, 0)

        res = []
        for level in range(min_v_order, max_v_order + 1):
            res.append(sorted(level_order[level]))

        return res

# Leetcode accepted
class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        columnTable = collections.defaultdict(list)
        min_column = max_column = 0

        def DFS(node, row, column):
            if node is not None:
                nonlocal min_column, max_column
                columnTable[column].append((row, node.val))
                min_column = min(min_column, column)
                max_column = max(max_column, column)

                # preorder DFS
                DFS(node.left, row + 1, column - 1)
                DFS(node.right, row + 1, column + 1)

        # step 1). DFS traversal
        DFS(root, 0, 0)

        # step 2). extract the values from the columnTable
        ret = []
        for col in range(min_column, max_column + 1):
            # sort first by 'row', then by 'value', in ascending order
            ret.append([val for row, val in sorted(columnTable[col])])

        return ret




# Problem 8 - Top View of Binary Tree
"""
apply the vertical order traversal logic (Problem 7) and select the first elemnt in the level order
"""




# Problem 9 - Bottom View of Binary Tre
"""
it's the last element in the vertical order traversal
==> keep replacing the node in the hashmap
"""




# Problem 10 - Right/Left View of Binary Tree
"""
It's the first/last element in the level order traversal
"""
# a small recursive function is as follows
res = []
def dfs(node, level):
    if not node:
        return
    if level == len(res):
        res.append(node.val)

    # for right view
    dfs(node.right)
    dfs(node.left)

    # for left view
    # dfs(node.left)
    # dfs(node.right)




# Problem 11 - Symmetric Tree
"""
https://leetcode.com/problems/symmetric-tree/
Given the root of a binary tree, check whether it is a mirror of itself 
(i.e., symmetric around its center).
"""
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        def checkSymmetric(p, q):
            if not p or not q:
                return p == q

            return p.val == q.val and checkSymmetric(p.left, q.right) and checkSymmetric(p.right, q.left)

        return checkSymmetric(root.left, root.right)




# Problem 12 - Print Root to Node Path in Binary Tree
"""
Give a node value return path from root to the node.
"""
def nodePath(root, node_val):

    path = []
    def dfs(node, target):
        if not node:
            return False

        # append the node
        path.append(node.val)

        # check if the current node is the target
        if node.val == target:
            return True
        
        # check if the children has the target node
        if dfs(node.left) or dfs(node.right):
            return True

        # since neither the node nor its childrens have the target node 
        # remove the node from the path 
        path.pop()

        return False
        
    dfs(root, node_val)
    return path




# Problem 13 - Lowest Common Ancestor of a Binary Tree
"""
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
"""
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node, p, q):
            if not node:
                return None

            if node.val == p.val or node.val == q.val:
                return node

            left_node = dfs(node.left, p, q)
            right_node = dfs(node.right, p, q)

            if not left_node:
                return right_node
            
            if not right_node:
                return left_node

            return node

        return dfs(root, p, q)




# Problem 14 - Maximum Width of Binary Tree
"""
* * Good question
https://leetcode.com/problems/maximum-width-of-binary-tree/
Given the root of a binary tree, return the maximum width of the given tree.
"""
# Solution :  
# apply level order but in the following fasion
#                   i  = (i - min index val at a particular level -> u can get this from queue 1st elem)
#                 /   \  
#          (2*i + 1)  (2*i + 2)  
#
# width  = last_index - first_index + 1  


class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        queue = collections.deque()
        queue.append((root, 0)) # <node, level>

        max_width = 0
        while queue:
            first_index, last_index = queue[0][1], queue[-1][1]
            width = last_index - first_index + 1
            max_width = max(width, max_width)

            min_level = queue[0][1]
            q_len = len(queue)

            for i in range(q_len):
                node, idx = queue.popleft()
                idx = idx - min_level
                if node.left:
                    queue.append((node.left, (idx * 2) + 1))
                if node.right:
                    queue.append((node.right, (idx * 2) + 2))

        return max_width




# Problem 15 - All Nodes Distance K in Binary Tree
"""
Given the root of a binary tree, the value of a target node target, and an integer k, 
return an array of the values of all nodes that have a distance k from the target node.

You can return the answer in any order.

Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2
Output: [7,4,1]
"""
# Solution : 
# since its a tree the egdes are uni-directional and we don't have any means to reach to 
# the parent node 
# so we will create a adjacency list such that while traversing the node we also add its parent
# 
# and thus this problem can be converted to graph problem and the traverse the graph (BFS) with
# target node as start node and return nodes which are at level 2




# Problem 16 - Construct Binary Tree from Preorder and Inorder Traversal
"""
https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
"""
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        p_idx = 0

        def dfs(i, j):
            nonlocal p_idx

            if i > j:
                return None

            root = TreeNode(preorder[p_idx])

            # finding the root node in in_order
            i_idx = 0
            for k in range(i, j + 1):
                if inorder[k] == preorder[p_idx]:
                    i_idx = k
                    break
            
            p_idx += 1
            
            root.left = dfs(i, i_idx - 1)
            root.right = dfs(i_idx + 1, j)
            
            return root
        
        return dfs(0, len(preorder) - 1)




# Problem 17 - Serialize and Deserialize Binary Tree
"""
https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
"""        
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(res)
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        ser_tree = data.split(",")
        self.i = 0

        def dfs():
            if ser_tree[self.i] == "N":
                self.i += 1
                return None

            tmp = TreeNode(int(ser_tree[self.i]))
            self.i += 1
            tmp.left = dfs()
            tmp.right = dfs()
            return tmp
        return dfs()




# Problem 18 - Flatten Binary Tree to Linked List
"""
* * Good question
https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
"""
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        prev = None

        def flattenUtil(node):
            nonlocal prev
            if not node:
                return

            flattenUtil(node.right)
            flattenUtil(node.left)

            node.right = prev
            node.left = None
            prev = node
            

        flattenUtil(root)




# Problem - Flatten a Multilevel Doubly Linked List
"""
https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/description/
same as flatten a tree
"""
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

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




# Problem 19 - Count Good Nodes in Binary Tree
"""
https://leetcode.com/problems/count-good-nodes-in-binary-tree/
Given a binary tree root, a node X in the tree is named good if in the path from root to X 
there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.

https://www.youtube.com/watch?v=7cp5imvDzl4&list=PLot-Xpze53ldg4pN6PfzoJY7KsKcxF1jg&index=3
"""
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        good_node_cnt = 0

        # Pre-order traversal
        def dfs(node, prev_max):
            nonlocal good_node_cnt

            # if the node val >= prev max seen so far
            if node.val >= prev_max:
                good_node_cnt += 1
                
            if node.left:
                dfs(node.left, max(node.val, prev_max))
            if node.right:
                dfs(node.right, max(node.val,prev_max))

        dfs(root, root.val)
        return good_node_cnt




# Problem 20 - Invert Binary Tree
"""
https://leetcode.com/problems/invert-binary-tree/
Given the root of a binary tree, invert the tree, and return its root.

i.e swap left and right child
"""
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def inversion(node):
            if not node:
                return 
            
            node.left, node.right = node.right, node.left
            inversion(node.left)
            inversion(node.right)

        inversion(root)
        return root




# Problem 21 - Merge Two Binary Trees
"""
https://leetcode.com/problems/merge-two-binary-trees/
You are given two binary trees root1 and root2.
Imagine that when you put one of them to cover the other, 
some nodes of the two trees are overlapped while the others are not.

Return the merged tree.

Note: The merging process must start from the root nodes of both trees.

Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
"""        
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root1, root2):
            if not root1 and not root2:
                return None
            val = 0
            val += root1.val if root1 else 0
            val += root2.val if root2 else 0

            left = dfs(root1.left if root1 else None, root2.left if root2 else None)
            right = dfs(root1.right if root1 else None, root2.right if root2 else None)

            return TreeNode(val, left, right)

        return dfs(root1, root2)





# Problem 22 - House Robber III
"""
https://leetcode.com/problems/house-robber-iii
Besides the root, each house has one and only one parent house. 
After a tour, the smart thief realized that all houses in this place form a binary tree. 
It will automatically contact the police if two directly-linked houses were broken into on 
the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob 
without alerting the police.

Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
"""
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # post-order
        def dfs(node):
            if not node:
                return [0,0]

            lf = dfs(node.left)
            rt = dfs(node.right)

            withRoot = node.val + lf[1] + rt[1]
            withoutRoot = max(lf) + max(rt)

            return [withRoot, withoutRoot]

        return max(dfs(root))




# Problem 23 - Flip Equivalent Binary Trees
"""
https://leetcode.com/problems/flip-equivalent-binary-trees/
For a binary tree T, we can define a flip operation as follows: 
choose any node, and swap the left and right child subtrees.

A binary tree X is flip equivalent to a binary tree Y 
if and only if we can make X equal to Y after some number of flip operations.

Given the roots of two binary trees root1 and root2, 
return true if the two trees are flip equivalent or false otherwise.
"""
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def isValidFlip(r1, r2):
            if not r1 or not r2:
                return not r1 and not r2

            if r1.val != r2.val:
                return False

            non_flip = isValidFlip(r1.left, r2.left) and isValidFlip(r1.right, r2.right)
            flip = isValidFlip(r1.left, r2.right) and isValidFlip(r1.right, r2.left)
            return non_flip or flip

        return isValidFlip(root1, root2)




# Problem 24 - Operations on Tree
"""
https://leetcode.com/problems/operations-on-tree/
The data structure should support the following functions:

* Lock: Locks the given node for the given user and prevents other users from 
        locking the same node. You may only lock a node using this function if the node is unlocked.
* Unlock: Unlocks the given node for the given user. You may only unlock a node using this 
          function if it is currently locked by the same user.
* Upgrade: Locks the given node for the given user and unlocks all of its descendants 
           regardless of who locked it. You may only upgrade a node if all 3 conditions are true:
            1. The node is unlocked,
            2. It has at least one locked descendant (by any user), and
            3. It does not have any locked ancestors.

https://www.youtube.com/watch?v=qK4PtjrVD0U&list=PLot-Xpze53ldg4pN6PfzoJY7KsKcxF1jg&index=24
"""    




# Problem 25 - All Possible Full Binary Trees
"""
https://leetcode.com/problems/all-possible-full-binary-trees/
Given an integer n, return a list of all possible full binary trees with n nodes. 
Each node of each tree in the answer must have Node.val == 0.
Input: n = 7
Output: [[0,0,0,null,null,0,0,null,null,0,0],
[0,0,0,null,null,0,0,0,0],
[0,0,0,0,0,0,0],
[0,0,0,0,0,null,null,null,null,0,0],
[0,0,0,0,0,null,null,0,0]]
"""
class Solution:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        dp = {0: [], 1: [TreeNode(0)]}
        def backtrack(n):

            if n in dp:
                return dp[n]

            res = []
            for l in range(n):
                r = n - 1 - l
                lf, rt = backtrack(l), backtrack(r)

                for t1 in lf:
                    for t2 in rt:
                        res.append(TreeNode(0, t1, t2))
            dp[n] = res
            return res

        return backtrack(n)




# Problem 26 - Subtree of Another Tree
"""
https://leetcode.com/problems/subtree-of-another-tree/
Given the roots of two binary trees root and subRoot, 
return true if there is a subtree of root with the same structure 
and node values of subRoot and false otherwise.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSame(r1, r2):
            if not r1 or not r2:
                return not r1 and not r2

            return (r1.val == r2.val and 
                    isSame(r1.left, r2.left) and 
                    isSame(r1.right, r2.right))

        def dfs(node):
            if node is None:
                return False

            if isSame(node, subRoot):
                return True

            return dfs(node.left) or dfs(node.right)

        return dfs(root)




# Problem 27 - Populating Next Right Pointers in Each Node
"""
https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
Populate each next pointer to point to its next right node. 
If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.
"""
# This problem has two solutions
import collections 

class Solution:
    def connect(self, root: 'TreeNode') -> 'TreeNode':
        
        if not root:
            return root
        
        # Initialize a queue data structure which contains
        # just the root of the tree
        Q = collections.deque([root])
        
        # Outer while loop which iterates over 
        # each level
        while Q:
            
            # Note the size of the queue
            size = len(Q)
            
            # Iterate over all the nodes on the current level
            for i in range(size):
                
                # Pop a node from the front of the queue
                node = Q.popleft()
                
                # This check is important. We don't want to
                # establish any wrong connections. The queue will
                # contain nodes from 2 levels at most at any
                # point in time. This check ensures we only 
                # don't establish next pointers beyond the end
                # of a level
                if i < size - 1:
                    node.next = Q[0]
                
                # Add the children, if any, to the back of
                # the queue
                if node.left:
                    Q.append(node.left)
                if node.right:
                    Q.append(node.right)
        
        # Since the tree has now been modified, return the root node
        return root

# Solution 2
class Solution:
    def connect(self, root: 'TreeNode') -> 'TreeNode':
        
        if not root:
            return root
        
        # Start with the root node. There are no next pointers
        # that need to be set up on the first level
        leftmost = root
        
        # Once we reach the final level, we are done
        while leftmost.left:
            
            # Iterate the "linked list" starting from the head
            # node and using the next pointers, establish the 
            # corresponding links for the next level
            head = leftmost
            while head:
                
                # CONNECTION 1
                head.left.next = head.right
                
                # CONNECTION 2
                if head.next:
                    head.right.next = head.next.left
                
                # Progress along the list (nodes on the current level)
                head = head.next
            
            # Move onto the next level
            leftmost = leftmost.left
        
        return root 




# Problem 31 - Construct String from Binary Tree
"""
https://leetcode.com/problems/construct-string-from-binary-tree/
Given the root of a binary tree, construct a string consisting of parenthesis 
and integers from a binary tree with the preorder traversal way, and return it.

Example 1:
Input: root = [1,2,3,4]
Output: "1(2(4))(3)"

Example 2:
Input: root = [1,2,3,null,4]
Output: "1(2()(4))(3)"
"""     
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        res_str = []
        def dfs(node):
            if not node: 
                return

            res_str.append(str(node.val))

            if node.left:
                res_str.append("(")
                dfs(node.left)
                res_str.append(")")

            if not node.left and node.right:
                res_str.append("()")

            if node.right:
                res_str.append("(")
                dfs(node.right)
                res_str.append(")")
                
        dfs(root)
        return "".join(res_str)   




# Problem 32 - Path With Maximum Minimum Value
"""
https://leetcode.com/problems/path-with-maximum-minimum-value
Given an m x n integer matrix grid, return the maximum score of a path starting at (0, 0) and 
ending at (m - 1, n - 1) moving in the 4 cardinal directions.

The score of a path is the minimum value in that path.

For example, the score of the path 8 ??? 4 ??? 5 ??? 9 is 4.

Input: grid = [[5,4,5],[1,2,6],[7,4,6]]
Output: 4
"""
# Has 2 good solutions 
#  
# Solution 1 : BFS + PriorityQueue 
#  
# Suppose we start from the top-left cell, check its two neighbors, and then visit 
# the neighboring cell with the larger value. We can imagine that this newly visited 
# cell will have other neighboring cells. Once again, we can consider all cells that 
# neighbor the two visited cells and then visit the cell with the largest value. We can 
# repeat these steps until we reach the bottom-right cell. Now we have a path of visited 
# cells that connects the top-left cell to the bottom-right cell. Since, at each step, we 
# always picked the unvisited neighbor with the largest value, it is guaranteed that the 
# smallest value seen so far is the largest possible minimum value (the largest score) in 
# a valid path. 
"""
Algorithm

1. Initialize:
    an empty priority queue pqpqpq and put the top-left cell in.
    the status of all the cells as unvisited.
    the minimum value min_valmin\_valmin_val as the value of the top-left cell.
2. Pop the cell with the largest value from the priority queue, mark it as visited, 
   and update the minimum value seen so far.
3. Check if the current cell has any unvisited neighbors. If so, add them to the priority queue.
4. Repeat from step 2 until we pop the bottom-right cell from the priority queue. 
   Return the updated minimum value as the answer.
""" 

# Solution 2 : Union Find 
"""
Algorithm
1. Sort all the cells decreasingly by their values.  
   (flatten the matrix and sort it -- convert to  list)
2. Iterate over the sorted cells from the largest value, for each visited cell, check if it has 
   any 4-directionally connected visited neighbor cells, if so, 
   we use the union-find data structure to connect it with its visited neighbors.
3. Check if the top-left cell is connected with the bottom-right cell.
4. If so, return the value of the last visited cell.
5. Otherwise, repeat from the step 2.
""" 






# =======================================================================================================
# =================================== BINARY SEARCH TREE PROBLEMS =======================================
# =======================================================================================================

# Problem 19 - K-th Smallest/Largest Element in BST
"""
https://leetcode.com/problems/kth-smallest-element-in-a-bst/
"""
# Solution :  
# Approach 1 : Using pre-order traversal in BST, store the value in a list and
# return the k-th smallest element
# however this takes O(n) space

# Approach 2 : Do a pre-order traversal and keep the counter 
# when the counter reaches the k count return that element   




# Problem 20 - Validate Binary Search Tree
"""
https://leetcode.com/problems/validate-binary-search-tree/
"""
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def checkIsValid(node, left_bound, right_bound):
            if not node:
                return True

            # check if node is following the bound otherwise return False
            if node.val >= right_bound or node.val <= left_bound:
                return False

            return checkIsValid(node.left, left_bound, node.val) and checkIsValid(node.right, node.val, right_bound)

        return checkIsValid(root, -math.inf, math.inf)




# Problem 21 - Lowest Common Ancestor of a Binary Search Tree
"""
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
"""
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        curr = root

        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                return curr




# Problem 22 - Construct a BST from a preorder traversal

# Problem 23 - Inorder Successor in BST

# Problem 22 - Convert Sorted Array to Binary Search Tree
"""
https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
Input: nums = [-10,-3,0,5,9]
Output: [0,-3,9,-10,null,5]
"""        
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def buildBST(L, R):
            if L > R:
                return None

            mid = (L+R)//2
            val = nums[mid]
            lf = buildBST(L,mid-1)
            rt = buildBST(mid+1,R)

            return TreeNode(val, lf, rt)

        return buildBST(0, len(nums)-1)

# Problem 24 - Binary Search Tree Iterator
"""
* * Good Problem
https://leetcode.com/problems/binary-search-tree-iterator
Implement the BSTIterator class that represents an iterator over 
the in-order traversal of a binary search tree (BST):

1. BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. 
   The root of the BST is given as part of the constructor. 
   The pointer should be initialized to a non-existent number smaller than any element in the BST.

2. boolean hasNext() Returns true if there exists a number in the traversal to 
   the right of the pointer, otherwise returns false.

3. int next() Moves the pointer to the right, then returns the number at the pointer.

Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]
"""
# Solution : 
# Take a stack data structure
# 1. from root node add all the node (from left subtree) which are extreme left
# 2. when there is a next operation pop element of the queue and print and also 
#    look for if the node has right -> then all the node in the right tree which are extreme left 
#    add them in the stack
# 3. for has_next operation if the stack is not empty return true





# Problem 25 - Two Sum In BST
"""
https://leetcode.com/problems/two-sum-iv-input-is-a-bst/
Check if there exists a pair with Sum K
"""
# U need BST Iterartor done in Problem - 25 
# U need to implement next() and before() function and then use these two to find the
# pair 




# Problem 26 - Recover Binary Search Tree
"""
https://leetcode.com/problems/recover-binary-search-tree/
"""




# Problem 27 - Trim a Binary Search Tree
"""
https://leetcode.com/problems/trim-a-binary-search-tree/
Given the root of a binary search tree and the lowest and highest boundaries as low and high, 
trim the tree so that all its elements lies in [low, high].

Return the root of the trimmed binary search tree.
Input: root = [1,0,2], low = 1, high = 2
Output: [1,null,2]
"""
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root: return None 
        if root.val > high: 
            return self.trimBST(root.left, low, high) 
        if root.val < low: 
            return self.trimBST(root.right, low, high) 
            
        root.left = self.trimBST(root.left, low, high) 
        root.right = self.trimBST (root.right, low, high) 
        
        return root
    



# Problem 23 - Unique Binary Search Trees
"""
https://leetcode.com/problems/unique-binary-search-trees/
Number of structurally unique BST's

Answer
Formula:
C(0) = 1

C(n+1) = [2(2n + 1)/n + 2] x C(n)
"""




# Problem 28 - Convert BST to Greater Tree
"""
https://leetcode.com/problems/convert-bst-to-greater-tree/
Given the root of a Binary Search Tree (BST), convert it to a Greater Tree 
such that every key of the original BST is changed to the original key plus 
the sum of all keys greater than the original key in BST.
Input: root = [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
https://www.youtube.com/watch?v=7vVEJwVvAlI&list=PLot-Xpze53ldg4pN6PfzoJY7KsKcxF1jg&index=35
"""
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        sum = 0
        def dfs(node):
            nonlocal sum
            # base condition
            if not node:
                return

            dfs(node.right)
            sum += node.val
            node.val = sum
            dfs(node.left)
        dfs(root)
        return root