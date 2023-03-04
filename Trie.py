"""
Implementing Trie or Prefix-tree

https://leetcode.com/problems/implement-trie-prefix-tree/

A trie (pronounced as "try") or prefix tree is a tree data structure 
used to efficiently store and retrieve keys in a dataset of strings. 
There are various applications of this data structure, such as autocomplete 
and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
1. void insert(String word) Inserts the string word into the trie.
2. boolean search(String word) 
    Returns true if the string word is in the trie (i.e., was inserted before), 
            and false otherwise.
3. boolean startsWith(String prefix) 
    Returns true if there is a previously inserted string word that has the prefix prefix, 
            and false otherwise.

Note : characters are from lower case a to z
"""

# solution : https://www.youtube.com/watch?v=oobqoCJlHA0

class TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.end_of_word = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root

        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
            
        curr.end_of_word = True

    def search(self, word: str) -> bool:
        curr = self.root

        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]

        return curr.children.end_of_word

    def startsWith(self, prefix: str) -> bool:
        curr = self.root

        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]

        return True
