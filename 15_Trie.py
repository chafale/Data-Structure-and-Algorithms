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
    



# Problem 2 - Design Add and Search Words Data Structure
"""
https://leetcode.com/problems/design-add-and-search-words-data-structure
Design a data structure that supports adding new words and finding if a s
tring matches any previously added string.

Implement the WordDictionary class:
1. WordDictionary() Initializes the object.
2. void addWord(word) Adds word to the data structure, it can be matched later.
3. bool search(word) Returns true if there is any string in the data structure 
   that matches word or false otherwise. word may contain dots '.' where dots can 
   be matched with any letter.

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]
"""
class TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.end_of_word = False
        
class WordDictionary:

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        curr = self.root

        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            
            curr = curr.children[char]
        
        curr.end_of_word = True

    def search(self, word: str) -> bool:

        def dfs(idx, root):
            curr = root

            for i in range(idx, len(word)):
                char = word[i]
                if char == ".":
                    for child in curr.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if char not in curr.children:
                        return False
                    curr = curr.children[char]

            return curr.end_of_word
        return dfs(0, self.root)
