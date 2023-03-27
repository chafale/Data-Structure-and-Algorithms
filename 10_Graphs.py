from ast import List
from collections import defaultdict
from collections import deque 
import heapq
import math
"""
Neetcode youtube series
https://www.youtube.com/playlist?list=PLot-Xpze53ldBT_7QA8NVot219jFNr_GI
"""

# Introduction
# Most common graphs algorithms
"""
1. DFS --> O(n) --> DataStruct: Stack, Hashset <to detect a cycle>
2. BFS --> O(n) --> DataStruct: Queue, Hashset
3. Union find : it is used to union together dis-joint sets & combine them together efficiently
    --> O(n log(n)) --> DataStruct: Random Forest
4. Topological Sort --> O(n) --> DataStruct: DFS
5. Dijkstra's Algo --> shortest path algorithm
    --> E log(V) --> DataStruct: Heap / Priority Queue
"""

graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

# Basics 1 - BFS Traversal
def BFS(graph, start_node):
    visited = set()
    queue = []

    visited.add(start_node)
    queue.append(start_node)

    while queue:
        tmp = queue.pop(0)
        # do something with the pop element -- u can just print the element

        for neighbour in graph[tmp]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
# Time complexity = O(n) + O(2E)




# Basics 2 - DFS Traversal
visited = set()
def DFS(node):
    if node not in visited:
        # do someting with the current node
        visited.add(node)
        for neighbour in graph[node]:
            DFS(neighbour)
# Time complexity = O(n) + O(2E)




# Basics 3 - Detect a cycle in an un-directed graph 
graph_cycle = {
    1 : [2, 3],
    2 : [1, 5],
    3 : [1, 4, 6],
    4 : [3],
    5 : [2, 7],
    6 : [3, 7],
    7 : [5, 6]
}
# Approach 1: using BFS
"""
Note : insert the node and its parent in the queue - <bcoz . . . . . .
and a graph has a cycle if the next insering node has already been visited
"""
def detectCycleBFS(start_node) -> bool:
    visited = set()
    queue = []

    queue.append((start_node, -1))
    visited.add(start_node)

    while queue:
        node, parent = queue.pop()

        for neighbour in graph_cycle[node]:
            if neighbour not in visited:
                queue.append((neighbour, node))
                visited.add(neighbour)

            # if neighbour is visited then check if its not the parent node as graph is un-directed
            elif neighbour != parent:
                # cycle has been detected
                return True
            
    return False

# Approach 2: using DFS
visited = set()
def detectCycleDFS(node, parent) -> bool:
    visited.add(node)

    for neighbour in graph_cycle[node]:
        if neighbour not in visited:
            if detectCycleDFS(neighbour, node) == True:
                return True
            elif neighbour != parent:
                return True

    return False




# Basics 4 - Detect a cycle in directed graph
directed_graph = {}
"""
    1 -> 2  -> 3   ->  4
         ^     |       |
         |     \/      \/
         8     7  ->   5  ->  6
         | ^
         |  \
         |   \
         |    \
         |     \
         \/     \
         9   -> 10

 * Please note here 3 - 4 - 5 - 7 is not a cycle 
 * but 8 - 9 - 10 is a cycle

 ** Here simple DFS call will not work
"""
# Approach 1 : using DFS
# ** Hint : on the same path the node has to be visited again 
#          -> here we have two visted data structure
#          -> 1. visited
#          -> 2. path_visited
#          when we back_track we remove the node from the path_visited

visited = set()
path_visited = set()

def DFS(node):
    visited.add(node)
    path_visited.add(node)

    for neighbour in directed_graph[node]:
        if neighbour not in visited:
            if DFS(neighbour) == True:
                return True
        elif neighbour in path_visited:
            return True

    # back_track we remove the node from the path_visited
    path_visited.remove(node)

    return False

# Approach 2 : using BFS : Kahn's algo : Topo sort
"""
Algorithm : topo sort is of n size
            so if u can't produce a topo sort of n size then there is a cycle in the graph

            Note : topo sort can only be applied on DAG, if not used on DAG topo sort 
            never converges --> we will use this property to find the cycle in the graph

 1. Apply topo sort
 2. if topo sort exactly has n elements then no cycle
 3. else cycle
"""
# copying topo-sort code from basic-6
def topoSort(graph):
    indegree = {n:0 for n in graph}
    computeIndegree(graph, indegree)

    queue = []
    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)

    """ Count variable to find the number of elements """
    count = 0

    while queue:
        tmp = queue.pop(0)

        """ count """
        count += 1

        for neighbour in graph[tmp]:
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)

    """ count """
    if count == len(graph):
        return False

    return True




# Basics 5 - Bipartite Graph
"""
If u can color the graph using two color such that no two
adjacent node have the same color -- Bipartite property
"""
# Note
"""
* Linear Graph + Even length cycle graph are all ==> Bipartite
* Odd lenght cycle graph ==> non Bipartite
"""
odd_graph_cycle = {
    1 : [2],
    2 : [1, 3, 6],
    3 : [2, 4],
    4 : [3, 5, 7],
    5 : [4, 6],
    6 : [2, 5],
    7 : [4, 8],
    8 : [7]
}
# Approach 1 : BFS <Bipartite>
def toggelColor(color):
    if color == 0:
        return 1
    return 0

def detectBipartiteBFS(start_node):
    visited = {} # key value pair <node : color>
    queue = []

    # Assuming starting node of color = 0
    visited[start_node] = 0
    queue.append(start_node)

    while queue:
        base = queue.pop(0)
        base_color = visited[base]

        for neighbour in odd_graph_cycle[base]:
            if neighbour not in visited:
                visited[neighbour] = toggelColor(base_color)
                queue.append(neighbour)
            
            elif visited[neighbour] == base_color:
                return False

    return True

# Approach 2 : DFS <Bipartite>
visited = {} # key value pair <node : color>
def detectBipartiteDFS(node, color):
    visited[node] = color

    for neighbour in odd_graph_cycle[node]:
        if neighbour not in visited:
            if detectBipartiteDFS(neighbour, toggelColor(color)) == False:
                return False
        
        elif visited[neighbour] == color:
            return False

    return True




# Basics 6 - Topological Sort
"""
Linear ordering of the vertices such that if there is an edge between 
u & v, u appears before v in the ordering

** Topological sort only works on DAG - Directed Acyclic Graph
"""
# Approach 1 : using DFS
# We will use stack data structure to store the topo-sort node
# ** whenever the all dfs call for the node is completed --> add node in the stack

visited = set()
topo_stack = []

def dfs(node):
    visited.add(node)

    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs(node)
    
    topo_stack.append(node)

def topoSort(graph):
    for node in graph:
        if node not in visited:
            dfs(node)

    return topo_stack[::-1]

# Approach 2 : KHAN's ALGORITHM -- BFS Algorithm
"""
Concept of indegree
* Indegree : no. of incoming edges in the node
when indegree of node is zero add them in the queue
and process elements in the queue and the order in which the elements 
are removed is the correct topo sort order
"""
# code to calculate the in-degree
def computeIndegree(graph, indegree):
    for node in graph:
        for neighbour in graph[node]:
            indegree[neighbour] += 1

def topoSort(graph):
    indegree = {n:0 for n in graph}
    computeIndegree(graph, indegree)

    queue = []
    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)

    topo_sort = []

    while queue:
        tmp = queue.pop(0)
        topo_sort.append(tmp)

        for neighbour in graph[tmp]:
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)

    return topo_sort




# Basic 7 : Dijkstras Algorithm
"""
Shortest path from source node to all the nodes

* * Dijkstras does not work when the graph has negative weight cycle
"""
def dijkstras(graph: List[List[int]], start_node):
    # create a adjacency list
    adj_list = defaultdict(dict)
    for frm, to, cost in graph:
        adj_list[frm][to] = cost

    distance = {node : float("inf") for node in adj_list.keys}
    distance[start_node] = 0

    heap = [(0, start_node)]
    visited = set()

    while heap:
        parent_dist, parent_node = heapq.heappop(heap)
        if parent_node not in visited:
            visited.add(parent_node)

            for neighbour in adj_list[parent_node]:
                if neighbour not in visited:
                    tot_dist = parent_dist + adj_list[parent_node][neighbour]
                    if tot_dist < distance[neighbour]:
                        distance[neighbour] = tot_dist
                        heapq.heappush(heap, (tot_dist, neighbour))

    if len(visited) != len(distance):
        return -1

    return distance




# Basic 8 : Bellman Ford Algorithm
"""
This algo is used to find the shortest path in the graph from src to all nodes

Why need of this algo when there is Dijkstras ?
Ans : 1. Dijkstras algo fails when graph has negative edges
      2. For negative edge cycle Dijkstras will fail
      3. Does not help to detect negative cycle

Bellman Ford Algorithm
1. shortest path
2. helps to detect negative edge cycle

Note : Bellman Ford Algorithm is only applicable to the directed graph
we can convert a un-directed graph to directed graph by showing two uni-directional edges

    u - v  =>  u -> v
                 <-
"""
# Algorithm :
#   Relax all the edges n-1 times
#   Relaxation : u -> v & edge wt = w then
#                if dist[u] + wt < dist[v]:
#                      dist[v] = dist[u] + wt

"""
Detecting negative edge cycle:
Relax all the edges one more time i.e the Nth time <last time>
if the distance matrix wts. changes then there is negative edge cycle
"""




# Basic 9 : Floyd Warshall Algorithm | Multi-source shortest path algorithm
"""
* * All source shortest path algorithm
Also helps to detect negative edge cycles

All previously seen were single source shortest path algorithm
"""
# Dynamic Programming
# where n : number of vertices
n = 6

# 1. create a adjacency matrix
cost = [[math.inf]*n for i in range(n)]

# 2. cost[i][i] = 0
for i in range(n):
    cost[i][i] = 0

# 3. apply dp
for via in range(0, n):
    for i in range(0, n):
        for j in range(0, n):
            if cost[i][via] == math.inf or cost[via][j] == math.inf: 
                continue
            cost[i][j] = min(cost[i][j], cost[i][via] +  cost[via][j])
"""
Detecting negative edge cycle:
if the cost of itself is not zero then there is a negative edge present

for i=0 -> n:
    if cost[i][i] < 0:
        "negative cycle"
"""




# Basic 10 - Minimum spanning tree
"""
Spanning Tree : A tree in which has 'n' nodes and 'n-1' edges & all nodes are
                reachable from each other

Minimum spanning tree : any spanning tree with least sum of weights 

Algorithms:
    1. Prims Algorithms
    2. Kruskal Algorithm
"""




# Basic 11 - Prims Algorithm
"""
DS used : Priority Queue < weight, node, parent >
          visited set
"""
def primsAlgo(graph):
    # priority queue
    
    visited = set()
    min_spanning_tree = list()
    pq = [(0, 0, -1)] # < weight, node, parent >

    mst_sum = 0
    while pq:
        wt, node, parent = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            if parent != -1:
                min_spanning_tree.append((node, parent))
            
            mst_sum += wt

            for neighbour in graph[node]:
                if neighbour not in visited:
                    heapq.heappush(pq, (neighbour, node))

    return min_spanning_tree, mst_sum




# Basic 12 - Disjoint Set
"""
Given a graph like :
    1 - 2 - 3 ; 4 - 5 ; 6 - 7
    
    Disjoint set will tell in constant time whether node 1 & node 4 belongs to same component or not

    Has 2 methods:
        1. findParent(node)
        2. Union()
"""
class DisjointSet:
    """
        Disjoint set using rank
    """
    def __init__(self, nodes) -> None:
        self.parent = [i for i in range(nodes)]
        self.rank = [1 for i in range(nodes)]


    def find_parent(self, node):
        if node == self.parent[node]:
            return node

        # path compression
        self.parent[node] = self.find_parent(self.parent[node])
        return self.parent[node]

    def union_by_rank(self, node1, node2) -> bool:
        p1 = self.find_parent(node1)
        p2 = self.find_parent(node2)

        # cannot union
        if p1 == p2:
            return False

        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += 1
        else:
            self.parent[p1] = p2
            self.rank[p2] += 1

        return True




# Problem 1 - Count the number of Connected Components in the graph
"""
1 - 2    4 - 5    7 - 8
    |        |
    3        6
"""
visited = set()
def dfs(node):
    if node not in visited:
        visited.add(node)
        for neighbour in graph[node]:
            dfs(neighbour)

def countConnectedComponents(graph):
    conn_component = 0
    for node in graph:
        if node not in visited:
            conn_component += 1
            dfs(node)
    return conn_component




# Problem 2 - Number of Islands
"""
https://leetcode.com/problems/number-of-islands/
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally 
or vertically. You may assume all four edges of the grid are all surrounded by water.

Example:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
"""
# Solution : same as finding connected components in the graph
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        r, c = len(grid), len(grid[0])

        visited = set()

        def dfs(i, j):
            if (i, j) not in visited:
                visited.add((i, j))

                # run dfs on neighbours
                if 0 <= i - 1 < r and grid[i-1][j] == "1":
                    dfs(i-1, j)
                if 0 <= i + 1 < r and grid[i+1][j] == "1":
                    dfs(i+1, j)
                if 0 <= j - 1 < c and grid[i][j-1] == "1":
                    dfs(i, j-1)
                if 0 <= j + 1 < c and grid[i][j+1] == "1":
                    dfs(i, j+1)

        islands = 0
        for i in range(r):
            for j in range(c):
                if grid[i][j] == "1" and (i, j) not in visited:
                    islands += 1
                    dfs(i, j)
        return islands




# Problem 3 - Flood Fill
"""
https://leetcode.com/problems/flood-fill/
You are also given three integers sr, sc, and color.
You should perform a flood fill on the image starting from the pixel image[sr][sc].
Example:
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
"""
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        r, c = len(image), len(image[0])
        def bfs(sr, sc, color):
            visited = set()
            q = []

            visited.add((sr, sc))
            q.append((sr, sc))

            start_color = image[sr][sc]

            while q:
                i, j = q.pop(0)
                image[i][j] = color

                # push all neighbours of (i, j) on queue
                if 0 <= i-1 and image[i-1][j] == start_color and (i-1,j) not in visited:
                    visited.add((i-1, j))
                    q.append((i-1, j))

                if i+1 < r and image[i+1][j] == start_color and (i+1,j) not in visited:
                    visited.add((i+1, j))
                    q.append((i+1, j))

                if 0 <= j-1 and image[i][j-1] == start_color and (i,j-1) not in visited:
                    visited.add((i, j-1))
                    q.append((i, j-1))

                if j+1 < c  and image[i][j+1] == start_color and (i,j+1) not in visited:
                    visited.add((i, j+1))
                    q.append((i, j+1))      

        bfs(sr, sc, color)
        return image




# Problem 4 - Rotting Oranges
"""
https://leetcode.com/problems/rotting-oranges/
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
Return the minimum number of minutes that must elapse until no cell has a fresh orange. 
If this is impossible, return -1.

Example 1:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4

Example 2:
Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
"""
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        r, c = len(grid), len(grid[0])
        time_elapse = 0
        
        def bfs():
            nonlocal time_elapse, r, c
            q = []

            for i in range(r):
                for j in range(c):
                    if grid[i][j] == 2:
                        q.append((i, j, 0))

            while q:
                i, j, level = q.pop(0)
                time_elapse = max(time_elapse, level)

                # push all neighbours of (i, j) on queue
                if 0 <= i-1 and grid[i-1][j] == 1:
                    grid[i-1][j] = 2
                    q.append((i-1, j, level+1))

                if i+1 < r and grid[i+1][j] == 1:
                    grid[i+1][j] = 2
                    q.append((i+1, j, level+1))

                if 0 <= j-1 and grid[i][j-1] == 1:
                    grid[i][j-1] = 2
                    q.append((i, j-1, level+1))

                if j+1 < c  and grid[i][j+1] == 1:
                    grid[i][j+1] = 2
                    q.append((i, j+1, level+1)) 

        bfs()

        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    return -1

        return time_elapse




# Problem 5 - 01 Matrix
"""
https://leetcode.com/problems/01-matrix/
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
Output: [[0,0,0],[0,1,0],[1,2,1]]
"""
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        r, c = len(mat), len(mat[0])

        distance = [[0 for j in range(c)] for i in range(r)]
        visited = set()

        q = []

        # find all zeros and add them in the queue
        for i in range(r):
            for j in range(c):
                if mat[i][j] == 0:
                    visited.add((i, j))
                    q.append((i, j, 0))

        delta = [[-1, 0],[1, 0],[0, -1], [0, 1]]
        while q:
            i, j, dist = q.pop(0)
            distance[i][j] = dist

            # iterate over all the neighbours of (i, j)
            for x, y in list(map(lambda d: (i + d[0], j + d[1]), delta)):
                if 0 <= x < r and 0 <= y < c and (x, y) not in visited:
                    visited.add((x, y))
                    q.append((x, y, dist+1))
        
        return distance




# Problem 6 - Surrounded Regions
"""
https://leetcode.com/problems/surrounded-regions/
Given a matrix mat of size N x M where every element is either `O` or `X`.
Replace all `O` with `X` that are surrounded by `X`.
A `O` (or a set of `O`) is considered to be by surrounded by 
`X` if there are `X` at locations just below, just above, just left and just right of it.
https://www.youtube.com/watch?v=BtdgAys4yMk&list=PLgUwDviBIf0oE3gA41TKO2H5bHpPd7fzn&index=14
"""

# Algorithm : start from the boundry zeros and mark them as visited 
# run dfs to find all connected zeros attached to boundry zeros and do not convert them
# convert the rest of all inner un-visted zeros to X

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        visited = set()
        r, c = len(board), len(board[0])

        delta = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        def dfs(i, j):
            visited.add((i, j))
            # find the neighbouring "O"
            for x, y in list(map(lambda x: (i+x[0], j+x[1]), delta)):
                if 0 <= x < r and 0 <= y < c and board[x][y] == "O" and (x, y) not in visited:
                    dfs(x, y)

        for i in range(r):
            if board[i][0] == "O":
                dfs(i, 0)
            if board[i][c-1] == "O":
                dfs(i, c-1)

        for j in range(c):
            if board[0][j] == "O":
                dfs(0, j)
            if board[r-1][j] == "O":
                dfs(r-1, j)

        for i in range(r):
            for j in range(c):
                if board[i][j] == "O" and (i, j) not in visited:
                    board[i][j] = "X"




# Problem 7 - Number of Enclaves
"""
https://leetcode.com/problems/number-of-enclaves/
"""
# sol : same as above




# Problem 8 - Number of Distinct Islands
"""
https://leetcode.com/problems/number-of-distinct-islands/
You are given an m x n binary matrix grid. 
An island is a group of 1's (representing land) connected 4-directionally 
(horizontal or vertical.) 
You may assume all four edges of the grid are surrounded by water.
Return the number of distinct islands.
"""
# How to store the shape of the island :
# consider (0,0) - (0,1)
#            |
#          (1,0)
# if we consider (0,0) as base
#
# shape : (0,0),(0,1),(1,0)
#
# and consider another structure as
# (2,2) - (2,3)
#   |
# (3,2)
# and (2,2) as base
# 
# if we do 
# (2,2) - (2,2) = (0,0)     |
# (2,3) - (2,2) = (0,1)     |   - we can compare with (0,0),(0,1),(1,0)
# (3,2) - (2,2) = (1.0)     |   - which is equal 
#
# 
# i.e if we do base_coor - coor we get shape 
#

class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        visited = set()
        unique_island = set()

        def dfs(i, j, base_row, base_col, current_island):
            if i < 0 or i >= r or j < 0 or j >= c:
                return
            
            if grid[i][j] == 0 or (i, j) in visited:
                return

            visited.add((i, j))
            current_island.add((i - base_row, j - base_col))

            dfs(i+1, j, base_row, base_col, current_island)
            dfs(i-1, j, base_row, base_col, current_island)
            dfs(i, j+1, base_row, base_col, current_island)
            dfs(i, j-1, base_row, base_col, current_island)


        r, c = len(grid), len(grid[0])
        for i in range(r):
            for j in range(c):

                # this is important step
                base_row, base_col = i, j

                current_island = set()
                dfs(i, j, base_row, base_col, current_island)
                if current_island:
                    unique_island.add(frozenset(current_island))

        return len(unique_island)




# Problem 9 - Eventual Safe States
"""
https://leetcode.com/problems/find-eventual-safe-states/
A node is a terminal node if there are no outgoing edges. 
A node is a safe node if every possible path starting from that node 
leads to a terminal node (or another safe node).

Return an array containing all the safe nodes of the graph. 
The answer should be sorted in ascending order.
"""
# Approach 1 : using DFS
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        visited = set()
        path_visited = set()

        def dfs(node):
            visited.add(node)
            path_visited.add(node)

            for neighbour in graph[node]:
                if neighbour not in visited:
                    if dfs(neighbour) == True:
                        return True
                # cycle detected
                elif neighbour in path_visited:
                    return True

            safe_node.append(node)
            path_visited.remove(node)
            return False
        
        safe_node = []
        for node, _ in enumerate(graph):
            if node not in visited:
                dfs(node)
        
        return sorted(safe_node)

# Approach 2 : using BFS | Topo-sort
"""
1. Reverse all the edges in the graph 
2. apply topo-sort
"""
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        rev_adj = {i:[] for i in range(len(graph))}
        # reverse the edges and calculate indegree
        indegree = {i:0 for i in range(len(graph))}

        for i, neighbours in enumerate(graph):
            for neighbour in neighbours:
                rev_adj[neighbour].append(i)
                indegree[i] += 1

        queue = []
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        safe_node = []
        while queue:
            tmp = queue.pop()
            safe_node.append(tmp)

            for neighbour in rev_adj[tmp]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)

        return sorted(safe_node)




# Problem 10 - Course Schedule I
"""
https://leetcode.com/problems/course-schedule/
There are a total of numCourses courses you have to take, 
labeled from 0 to numCourses - 1. You are given an array prerequisites.
Return true if you can finish all courses. Otherwise, return false.
"""
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # create adj-matrix
        adj_matrix = {i : [] for i in range(numCourses)}
        for elem in prerequisites:
            adj_matrix[elem[1]].append(elem[0])

        # apply topo sort
        # 1. find the indegree
        indegree = {i : 0 for i in range(numCourses)}
        for node in adj_matrix:
            for neighbour in adj_matrix[node]:
                indegree[neighbour] += 1

        # 2. topo-sort
        queue = []
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        count = 0

        while queue:
            tmp = queue.pop(0)
            count += 1

            for neighbour in adj_matrix[tmp]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
        
        if count == numCourses:
            return True

        return False




# Problem 11 - Course Schedule II
"""
https://leetcode.com/problems/course-schedule-ii/
Return the topo ordering
"""
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # create adj-matrix
        adj_matrix = {i : [] for i in range(numCourses)}
        for elem in prerequisites:
            adj_matrix[elem[1]].append(elem[0])

        # apply topo sort
        # 1. find the indegree
        indegree = {i : 0 for i in range(numCourses)}
        for node in adj_matrix:
            for neighbour in adj_matrix[node]:
                indegree[neighbour] += 1

        # 2. topo-sort
        queue = []
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        topo_order = []

        while queue:
            tmp = queue.pop(0)

            topo_order.append(tmp)

            for neighbour in adj_matrix[tmp]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)
        
        if len(topo_order) == numCourses:
            return topo_order

        return []




# Problem 12 - Alien Dictionary
"""
* * HARD PROBLEM * *
https://leetcode.com/problems/alien-dictionary/

Example 1: 
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
"""
# create a DAG and apply toposort
class Solution:
    def alienOrder(self, words: List[str]) -> str:

        # create a adj martix
        adj_matrix = {}
        for word in words:
            for char in word:
                if char not in adj_matrix:
                    adj_matrix[char] = []

        # create DAG
        for i in range(0, len(words)-1):
            # comparing string i and i+1
            str_len = min(len(words[i]), len(words[i+1]))
            for j in range(str_len):
                if words[i][j] != words[i+1][j]:
                    adj_matrix[words[i][j]].append(words[i+1][j])
                    break

        # calculate indegree
        indegree = {c:0 for c in adj_matrix}
        for c in adj_matrix:
            for neighbour in adj_matrix[c]:
                indegree[neighbour] += 1

        queue = []
        for node in indegree:
            if indegree[node] == 0:
                queue.append(node)

        dict_order = []

        while queue:
            tmp = queue.pop()
            dict_order.append(tmp)

            for neighbour in adj_matrix[tmp]:
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)

        if len(dict_order) == len(adj_matrix):
            return "".join(dict_order)
        return ""




# Problem 13 - Word Ladder I
"""
* * HARD PROBLEM * *
https://leetcode.com/problems/word-ladder/
Given two words, beginWord and endWord, and a dictionary wordList, 
return the number of words in the shortest transformation sequence 
from beginWord to endWord, or 0 if no such sequence exists.

Example 1:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: One shortest transformation sequence is 
"hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
""" 
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        res = 0
        wordSet = set(wordList)
        queue = []

        queue.append((beginWord, 1))
        if beginWord in wordSet: 
            wordSet.remove(beginWord)

        while queue:
            word, level = queue.pop(0)
        
            for i in range(len(word)):
                for num in range(97, 123):
                    new_word = word[:i] + chr(num) + word[i+1:]
                    if new_word in wordSet:
                        if new_word == endWord:
                            return level + 1
                        wordSet.remove(new_word)
                        queue.append((new_word, level + 1))

        return 0




# Problem 14 - Word Ladder II
"""
https://leetcode.com/problems/word-ladder-ii/
Given two words, beginWord and endWord, and a dictionary wordList, 
return all the shortest transformation sequences from beginWord to endWord, 
or an empty list if no such sequence exists. 
Each sequence should be returned as a list of the words [beginWord, s1, s2, ..., sk].
"""
# youtube solution
#  https://www.youtube.com/watch?v=AD4SFl7tu7I&list=PLgUwDviBIf0oE3gA41TKO2H5bHpPd7fzn&index=31




# Problem 15 - Shortest Path in Binary Matrix
"""
https://leetcode.com/problems/shortest-path-in-binary-matrix/
Given an n x n binary matrix grid, return the length of the shortest 
clear path in the matrix. If there is no clear path, return -1.
A clear path in a binary matrix is a path from the top-left cell to the 
bottom-right cell such that:
* All the visited cells of the path are 0.
* All the adjacent cells of the path are 8-directionally connected.
The length of a clear path is the number of visited cells of this path.
Example 1:
Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
"""
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        r, c = len(grid), len(grid[0])
        # source and destination loc
        sr, sc, dr, dc = 0, 0, r-1, c-1

        if grid[sr][sc] != 0 or grid[dr][dc] != 0:
            return -1

        q = [(sr, sc)]
        grid[sr][sc] = 1
        delta = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

        while q:
            _r, _c = q.pop(0)
            dist = grid[_r][_c]
            
            for x, y in list(map(lambda d: (_r + d[0], _c + d[1]), delta)):
                if 0 <= x < r and 0 <= y < c and grid[x][y] == 0:
                    grid[x][y] = dist + 1
                    q.append((x, y))
        return -1




# Problem 16 - Path With Minimum Effort
"""
https://leetcode.com/problems/path-with-minimum-effort/
You are situated in the top-left cell, (0, 0), and 
you hope to travel to the bottom-right cell, (rows-1, columns-1).
You can move up, down, left, or right, and you wish to find a route that 
requires the minimum effort.
A route's effort is the maximum absolute difference in heights between 
two consecutive cells of the route.
Return the minimum effort required to travel from the top-left cell to the bottom-right cell.
Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
Output: 2
"""
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        r, c = len(heights), len(heights[0])

        diff_matrix = [[math.inf]*c for _ in range(r)]
        heap = []

        diff_matrix[0][0] = 0
        heap.append((0, 0, 0))

        delta = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        while heap:
            diff, _r, _c = heapq.heappop(heap)

            for x, y in list(map(lambda d: (_r + d[0], _c + d[1]), delta)):
                if 0 <= x < r and 0 <= y < c:
                    curr_diff = abs(heights[x][y] - heights[_r][_c])
                    max_diff = max(curr_diff, diff)
                    if diff_matrix[x][y] > max_diff:
                        diff_matrix[x][y] = max_diff
                        heapq.heappush(heap, (max_diff, x, y))

        return diff_matrix[-1][-1]




# Problem 17 - Cheapest Flights Within K Stops
"""
https://leetcode.com/problems/cheapest-flights-within-k-stops/
You are also given three integers src, dst, and k, return the cheapest price 
from src to dst with at most k stops. If there is no such route, return -1.
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Output: 700
"""
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # create adjacency list
        flights_graph = defaultdict(dict)
        for frm, to, cost in flights:
            flights_graph[frm][to] = cost

        distance = {i:math.inf for i in range(n)}
        q = [(0, src, 0)]  #<k, node, cost>
        distance[src] = 0

        while q:
            _k, _node, _cost = q.pop(0)
            if _k > k:
                continue

            for neighbour in flights_graph[_node]:
                dist = _cost + flights_graph[_node][neighbour]
                if distance[neighbour] > dist and _k <= k:
                    distance[neighbour] = dist
                    q.append((_k + 1, neighbour, dist))

        return distance[dst] if distance[dst] != math.inf else -1




# Problem 18 - Number of Ways to Arrive at Destination
"""
https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
You want to know in how many ways you can travel from intersection 0 to 
intersection n - 1 in the shortest amount of time.
Return the number of ways you can arrive at your destination in the shortest amount of time. 
Since the answer may be large, return it modulo 109 + 7.
Input: n = 7, roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]
Output: 4
"""
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        intersects = defaultdict(dict)
        for frm, to, time in roads:
            intersects[frm][to] = time
            intersects[to][frm] = time

        distance = {i:math.inf for i in range(n)}
        ways = {i:0 for i in range(n)}

        distance[0] = 0
        ways[0] = 1
        pq = [(0, 0)]   # < distance, node>

        while pq:
            _dist, _node = heapq.heappop(pq)

            for neignbour in intersects[_node]:
                tot_time = _dist + intersects[_node][neignbour]
                if tot_time < distance[neignbour]:
                    distance[neignbour] = tot_time
                    heapq.heappush(pq, (tot_time, neignbour))
                    ways[neignbour] = ways[_node]
                elif tot_time == distance[neignbour]:
                    ways[neignbour] = ways[neignbour] + ways[_node]

        return ways[n-1] % (10**9 + 7)




# Problem 19 - Find the City With the Smallest Number of Neighbors at a Threshold Distance
"""
https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/
Return the city with the smallest number of cities that are reachable through some path 
and whose distance is at most distanceThreshold,
If there are multiple such cities, return the city with the greatest number.
Input: n = 4, edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], distanceThreshold = 4
Output: 3
Explanation: The figure above describes the graph. 
The neighboring cities at a distanceThreshold = 4 for each city are:
City 0 -> [City 1, City 2] 
City 1 -> [City 0, City 2, City 3] 
City 2 -> [City 0, City 1, City 3] 
City 3 -> [City 1, City 2] 
Cities 0 and 3 have 2 neighboring cities at a distanceThreshold = 4, 
but we have to return city 3 since it has the greatest number.
"""
# Floyd warshal algorithm
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:

        # create adjacency matrix
        cost = [[math.inf]*n for _ in range(n)]

        for frm, to, wt in edges:
            cost[frm][to] = wt
            cost[to][frm] = wt

        # cost[i][i] = 0
        for i in range(n):
            cost[i][i] = 0

        # Floyd warshal algorithm
        for via in range(n):
            for i in range(n):
                for j in range(n):
                    if cost[i][via] == math.inf or cost[via][j] == math.inf: 
                        continue
                    cost[i][j] = min(cost[i][j], cost[i][via] +  cost[via][j])

        city_no = -1
        cnt = n
        for i in range(n):
            curr_cnt = 0
            for j in range(n):
                if cost[i][j] <= distanceThreshold:
                    curr_cnt += 1

            if curr_cnt <= cnt:
                cnt = curr_cnt
                city_no = i
        
        return city_no




# Problem 19 - Pacific Atlantic Water Flow
"""
https://leetcode.com/problems/pacific-atlantic-water-flow/
Return a list of grid coordinates result where result[i] = [ri, ci] denotes 
that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

Hint :
Instead of looking for every path from cell to ocean, 
let's start at the oceans and try to work our way to the cells.
i.e lets start from the grid border 

we start traversing from the ocean -> check for higher height instead of lower height

"""
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        row, col = len(heights), len(heights[0])

        pacific = set()
        atlantic = set()

        def dfs(r, c, visit_set, prev_ht):
            if (r, c) in visit_set or r < 0 or c < 0 \
                or r == row or c == col or heights[r][c] < prev_ht:
                return
            
            visit_set.add((r, c))
            dfs(r + 1, c, visit_set, heights[r][c])
            dfs(r - 1, c, visit_set, heights[r][c])
            dfs(r, c + 1, visit_set, heights[r][c])
            dfs(r, c - 1, visit_set, heights[r][c])

        for c in range(col):
            dfs(0, c, pacific, heights[0][c])
            dfs(row - 1, c, atlantic, heights[row - 1][c])

        for r in range(row):
            dfs(r, 0, pacific, heights[r][0])
            dfs(r, col - 1, atlantic, heights[r][col - 1])

        result = []
        for r in range(row):
            for c in range(col):
                if (r, c) in pacific and (r, c) in atlantic:
                    result.append([r, c])

        return result             




# Problem 20 - Network Delay Time
"""
https://leetcode.com/problems/network-delay-time/
We will send a signal from a given node k. Return the minimum time it takes for all 
the n nodes to receive the signal. If it is impossible for all the n nodes to 
receive the signal, return -1.

Solve using Dijkstras
"""




# Problem 21 - Word Search
"""
Very Popular problem

https://leetcode.com/problems/word-search/
Given an m x n grid of characters board and a string word, return true if word exists in the grid.
"""
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        row, col = len(board), len(board[0])

        visited = set()

        def dfs(r, c, i):
            # return t/f
            if i == len(word):
                return True

            if (r, c) in visited or r <0  or c <0 or r >= row or c >= col or word[i] != board[r][c]:
                return False
            
            
            visited.add((r, c))
            res = (dfs(r+1, c, i+1) or dfs(r-1, c, i+1) or dfs(r, c+1, i+1) or dfs(r, c-1, i+1))
            
            visited.remove((r, c))
            return res
        
        for r in range(row):
            for c in range(col):
                if board[r][c] == word[0]:
                    if dfs(r, c, 0):
                        return True
                
        return False




# Problem 22 - Clone Graph
"""
https://leetcode.com/problems/clone-graph/
Return a deep copy (clone) of the graph.

concept is some-what un-common
"""
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        old_new = dict()

        def dfs(node):
            if node in old_new:
                return old_new[node]

            copy = Node(val=node.val)
            old_new[node] = copy
            for neighbour in node.neighbors:
                copy.neighbors.append(dfs(neighbour))

            return copy

        return dfs(node) if node else None





# Problem 23 - Island Perimeter
"""
https://leetcode.com/problems/island-perimeter/
Go through every cell on the grid and whenever you are at cell 1 (land cell), 
look for surrounding (UP, RIGHT, DOWN, LEFT) cells. 
A land cell without any surrounding land cell will have a perimeter of 4. 
Subtract 1 for each surrounding land cell.
"""
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        
        rows = len(grid)
        cols = len(grid[0])
        
        result = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    if r == 0:
                        up = 0
                    else:
                        up = grid[r-1][c]
                    if c == 0:
                        left = 0
                    else:
                        left = grid[r][c-1]
                    if r == rows-1:
                        down = 0
                    else:
                        down = grid[r+1][c]
                    if c == cols-1:
                        right = 0
                    else:
                        right = grid[r][c+1]
                        
                    result += 4-(up+left+right+down)
                
        return result
                



# Problem 24 - Graph Valid Tree
"""
https://leetcode.com/problems/graph-valid-tree/
You are given an integer n and a list of edges.
Return true if the edges of the given graph make up a valid tree, and false otherwise.
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Solving using : Union-find
"""
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        parent = [i for i in range(len(edges) + 1)]
        rank = [1 for i in range(len(edges) + 1)]

        def find_parent(n):
            if n == parent[n]:
                return n
            parent[n] = find_parent(parent[n])
            return parent[n]

        def union(n1, n2):
            p1 = find_parent(n1)
            p2 = find_parent(n2)

            if p1 == p2:
                return False

            if rank[p1] > rank[p2]:
                parent[p2] = p1
                rank[p1] += 1
            else:
                parent[p1] = p2
                rank[p2] += 1

            return True

        for u, v in edges:
            if not union(u, v):
                return False

        return True




# Problem 25 - Redundant Connection
"""
https://leetcode.com/problems/redundant-connection/
Return an edge that can be removed so that the resulting graph is a tree of n nodes. 
If there are multiple answers, return the answer that occurs last in the input.

Solving using union find
"""
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = [i for i in range(len(edges) + 1)]
        rank = [1 for i in range(len(edges) + 1)]

        def find_parent(n):
            if n == parent[n]:
                return n
            parent[n] = find_parent(parent[n])
            return parent[n]

        def union(n1, n2):
            p1 = find_parent(n1)
            p2 = find_parent(n2)

            if p1 == p2:
                return False

            if rank[p1] > rank[p2]:
                parent[p2] = p1
                rank[p1] += 1
            else:
                parent[p1] = p2
                rank[p2] += 1

            return True

        for u, v in edges:
            if not union(u, v):
                return [u, v]




# Problem 26 - Min Cost to Connect all Points
"""
https://leetcode.com/problems/min-cost-to-connect-all-points/

Apply prims algorithm
"""




# Problem 26 - Count Sub Islands
"""
* * Very Good Concept

https://leetcode.com/problems/count-sub-islands/
An island in grid2 is considered a sub-island if there is an island in grid1 
that contains all the cells that make up this island in grid2.

Return the number of islands in grid2 that are considered sub-islands.
"""
# Approach 1 : find all island in grid 2 and if those island are present in grid 1 increase the count
# this approach is pretty in-efficient

# Approach 2 : symultaneous dfs check on both grid 1 & grid 2

class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        row, col = len(grid1), len(grid1[0])
        visited = set()

        def dfs(r, c):
            # should return t/f
            if (r < 0 or c < 0 or r == row or c == col or 
                grid2[r][c] == 0 or (r, c) in visited):
                return True

            visited.add((r, c))

            
            res = True
            if grid1[r][c] == 0: 
                # at this condition we will not return bcoz we still want to make 
                # all the remaing connected cells of the island as visited so
                # that we don't visit them again 
                res = False

            res = dfs(r + 1, c) and res
            res = dfs(r - 1, c) and res
            res = dfs(r, c + 1) and res
            res = dfs(r, c - 1) and res
            return res

        count  = 0
        for r in range(row):
            for c in range(col):
                if grid2[r][c] == 1 and (r, c) not in visited and dfs(r, c):
                    count += 1
        return count




# Problem 27 - Swim in Rising Water
"""
* * HARD Problem
== GREEDY GRAPH PROBLEM ==

https://leetcode.com/problems/swim-in-rising-water/

You are given an n x n integer matrix grid.
You can swim from a square to another 4-directionally adjacent square 
if and only if the elevation of both squares individually are at most t.

Return the least time until you can reach the bottom right square (n - 1, n - 1) 
if you start at the top left square (0, 0).

Hint : use heap

https://www.youtube.com/watch?v=amvrKlMLuGY&list=PLot-Xpze53ldBT_7QA8NVot219jFNr_GI&index=19
"""
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # n x n integer matrix
        n = len(grid)

        visited = set()
        min_heap = [[grid[0][0], 0, 0]] # <max-height, row, col>
        visited.add((0, 0))

        delta = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while min_heap:
            ht, r, c = heapq.heappop(min_heap)

            if r == n-1 and c == n-1:
                return ht

            # neighbours
            for dx, dy in delta:
                x , y = r + dx, c + dy
                if (x < 0 or c < 0 or x >= n or y >= n or (x, y) in visited):
                    continue
                
                visited.add((x, y))
                # add max height so far seen in the path to the heap
                heapq.heappush(min_heap, [max(ht, grid[x][y]), x, y])





# Problem 28 - Walls and Gates
"""
https://leetcode.com/problems/walls-and-gates/

Given an m x n grid rooms:
-1 : walls
0 : gate
INF : empty rooms

Fill each empty room with the distance to its nearest gate. 
If it is impossible to reach a gate, it should be filled with INF.
INF = 2147483647

Hint : multi-souce BFS
"""
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        # Note : -1: walls, 0: gate, INF: empty room
        row, col = len(rooms), len(rooms[0])

        queue = []
        visited = set()

        # identify all gates and put them in the queue
        for r in range(row):
            for c in range(col):
                if rooms[r][c] == 0:
                    queue.append([0, r, c])
                    visited.add((r, c))

        delta = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while queue:
            dist, r, c = queue.pop(0)

            # for all neighbours of (r, c)
            for dx, dy in delta:
                x, y = r + dx, c + dy
                if (x < 0 or y < 0 or x >= row or y >= col or 
                    rooms[x][y] == -1 or (x, y) in visited):
                    continue

                rooms[x][y] = dist + 1
                visited.add((x, y))
                queue.append([dist + 1, x, y])




# Problem 29 : Max Area of Island
"""
https://leetcode.com/problems/max-area-of-island/
Return the maximum area of an island in grid. If there is no island, return 0.
"""
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])

        visited = set()

        def dfs(r, c):
            if (r < 0 or c < 0 or r >= row or c >= col 
                or (r, c) in visited or grid[r][c] != 1):
                return 0

            visited.add((r, c))
            area = 1 + dfs(r+1, c) + dfs(r-1, c) + dfs(r, c+1) + dfs(r, c-1)
            return area
        
        max_area = 0
        for r in range(row):
            for c in range(col):
                if grid[r][c] == 1 and (r, c) not in visited:
                    area = dfs(r, c)
                    max_area = max(max_area, area)
        return max_area




# Problem 30 : Reconstruct Itinerary
"""
* * HARD PROBLEM
https://leetcode.com/problems/reconstruct-itinerary/

Skiped for now
"""




# Problem 31 : Longest Increasing Path in a Matrix
"""
* * HARD Problem - but its doable
https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
Given an m x n integers matrix, return the length of the longest increasing path in matrix.

== can skip
"""
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        row, col = len(matrix), len(matrix[0])

        lip_matrix = [[-1 for _ in range(col)] for _ in range(row)]

        def dfs(r, c, prev):
            if r < 0 or c < 0 or r >= row or c >= col or matrix[r][c] <= prev:
                return 0

            if lip_matrix[r][c] != -1:
                return lip_matrix[r][c]
            
            res = 1
            res = max(res, 1 + dfs(r+1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r-1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r, c+1, matrix[r][c]))
            res = max(res, 1 + dfs(r, c-1, matrix[r][c]))

            lip_matrix[r][c] = res
            return res

        max_len = 0
        for r in range(row):
            for c in range(col):
                max_len = max(max_len, dfs(r, c, -1))
        return max_len





# Problem 32 : Open the Lock
"""
https://leetcode.com/problems/open-the-lock/

You have a lock in front of you with 4 circular wheels
The lock initially starts at '0000'
You are given a list of deadends dead ends

Given a target representing the value of the wheels that will unlock the lock, 
return the minimum total number of turns required to open the lock, 
or -1 if it is impossible.
"""
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        if "0000" in deadends:
            return -1

        def children(wheel):
            res = []
            for i in range(4):
                digit = str((int(wheel[i]) + 1) % 10)
                res.append(wheel[:i] + digit + wheel[i + 1 :])
                digit = str((int(wheel[i]) + 10 - 1) % 10)
                res.append(wheel[:i] + digit + wheel[i + 1 :])
            return res

        q = deque()
        visit = set(deadends)
        q.append(["0000", 0])  # [wheel, turns]
        while q:
            wheel, turns = q.popleft()
            if wheel == target:
                return turns
            for child in children(wheel):
                if child not in visit:
                    visit.add(child)
                    q.append([child, turns + 1])
        return -1




# Problem 33 : Shortest Bridge
"""
https://leetcode.com/problems/shortest-bridge/

You may change 0's to 1's to connect the two islands to form one island.
Return the smallest number of 0's you must flip to connect the two islands.

Note : here there are only two islands

Hint : multi-source BFS

Approach :
1. We will find 1st island using DFS 
2. Then we will use 1st island cells as multi-source BFS to find the 2nd island
"""
# https://www.youtube.com/watch?v=gkINMhbbIbU&list=PLot-Xpze53ldBT_7QA8NVot219jFNr_GI&index=29




# Problem 31 - Path With Maximum Minimum Value
"""
https://leetcode.com/problems/path-with-maximum-minimum-value
Given an m x n integer matrix grid, return the maximum score of a path starting at (0, 0) and 
ending at (m - 1, n - 1) moving in the 4 cardinal directions.

The score of a path is the minimum value in that path.

For example, the score of the path 8 → 4 → 5 → 9 is 4.

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




# Problem 32 - Accounts Merge
"""
https://leetcode.com/problems/accounts-merge/
Given a list of accounts, we would like to merge these accounts.
Two accounts definitely belong to the same person if there is some common email to both accounts.
Note that even if two accounts have the same name, they may belong to different people as people 
could have the same name.
A person can have any number of accounts initially, but all of their accounts definitely have the 
same name.

Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],
                    ["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],
                    ["John","johnnybravo@mail.com"]]

Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],
        ["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
"""
# Union Find 

class UnionFind:
    def __init__(self, n) -> None:
        self.parent = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    def union(self, node1, node2)-> bool:
        p1, p2  = self.find(node1), self.find(node2)

        if p1 == p2:
            return False
        
        if self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
            self.rank[p1] += 1
        else:
            self.parent[p1] = p2
            self.rank[p2] += 1
        return True


class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        union_find = UnionFind(len(accounts))

        email_2_account = {} # email : account_index

        for idx, account in enumerate(accounts):
            for email in account[1:]:
                if email in email_2_account:
                    union_find.union(idx, email_2_account[email])
                else:
                    email_2_account[email] = idx

        emailGroup = defaultdict(list)
        for email, idx in email_2_account.items():
            leader = union_find.find(idx)
            emailGroup[leader].append(email)

        res = []
        for idx, email in emailGroup.items():
            acc_name = accounts[idx][0]
            res.append([acc_name] + sorted(emailGroup[idx]))

        return res
    



# Problem 33 - Jump Game III
"""
Given an array of non-negative integers arr, you are initially positioned at 
start index of the array. When you are at index i, you can jump to i + arr[i] or i - arr[i], 
check if you can reach to any index with value 0.

Example 1:
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation: 
All possible ways to reach at index 3 with value 0 are: 
index 5 -> index 4 -> index 1 -> index 3 
index 5 -> index 6 -> index 4 -> index 1 -> index 3 

Example 2:
Input: arr = [4,2,3,0,3,1,2], start = 0
Output: true 
Explanation: 
One possible way to reach at index 3 with value 0 is: 
index 0 -> index 4 -> index 1 -> index 3

Example 3:
Input: arr = [3,0,2,1,2], start = 2
Output: false
Explanation: There is no way to reach at index 1 with value 0.
"""
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        q = deque()
        q.append([start, arr[start]])
        visited = set()
        visited.add(start)

        while q:
            idx, val = q.popleft()
            if val == 0:
                return True
            
            rt_jump = idx + val
            lf_jump = idx - val

            if rt_jump < len(arr) and (rt_jump) not in visited:
                q.append([rt_jump, arr[rt_jump]])
                visited.add(rt_jump)

            if lf_jump >= 0 and (lf_jump) not in visited:
                q.append([lf_jump, arr[lf_jump]])
                visited.add(lf_jump)

        return False
    



# Problem 34 - Minimum Score of a Path Between Two Cities
"""
https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities
The score of a path between two cities is defined as the minimum distance of a road in this path.
Return the minimum possible score of a path between cities 1 and n.

Example 1:
Input: n = 4, roads = [[1,2,9],[2,3,6],[2,4,5],[1,4,7]]
Output: 5
Explanation: The path from city 1 to 4 with the minimum score is: 1 -> 2 -> 4. 
The score of this path is min(9,5) = 5.
It can be shown that no other path has less score.

Example 2:
Input: n = 4, roads = [[1,2,2],[1,3,4],[3,4,7]]
Output: 2
Explanation: The path from city 1 to 4 with the minimum score is: 1 -> 2 -> 1 -> 3 -> 4. 
The score of this path is min(2,2,4,7) = 2.
"""
# Solution : sol to this problem is quite simple -- we just need to find the edge with min cost
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        # implementing dijkstras algorithm
        adj_list = defaultdict(dict)
        for u, v, dist in roads:
            adj_list[u][v] = dist
            adj_list[v][u] = dist
        
        visited = set()
        res = 1000000

        def dfs(node):
            nonlocal res

            if node in visited:
                return
            
            visited.add(node)
            for neighbour, dist in adj_list[node].items():
                res = min(res, dist)
                dfs(neighbour)

        dfs(1)
        return res