# 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。

# 示例:

# Trie trie = new Trie();

# trie.insert("apple");
# trie.search("apple");   // 返回 true
# trie.search("app");     // 返回 false
# trie.startsWith("app"); // 返回 true
# trie.insert("app");   
# trie.search("app");     // 返回 true
# 说明:

# 你可以假设所有的输入都是由小写字母 a-z 构成的。
# 保证所有输入均为非空字符串。
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # {
        #     'a':{
        #         'p':{
        #             'p':{
        #                 'l':{
        #                     'e':{'\0':{}}
        #                 },
        #                '\0':{}
        #             }
        #         }
        #     }
        # }
        self.trie = dict()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        curr = self.trie
        for i,ch in enumerate(word):
            if curr.get(ch, {}):
                curr = curr[ch]
                if i==len(word)-1:
                    curr['\0'] = {}
            else:
                if i<len(word)-1:
                    curr[ch] = {}
                    curr = curr[ch]
                else:
                    curr[ch] = {
                        '\0':{}
                    }

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self.trie
        for i, ch in enumerate(word):
            curr = curr.get(ch, {})
            if curr:
                continue
            else:
                break
        
        if i==len(word)-1 and '\0' in curr:
            ret = True
        else:
            ret = False

        return ret
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        ret = True
        curr = self.trie
        for i, ch in enumerate(prefix):
            curr = curr.get(ch, {})
            if curr:
                continue
            else:
                break
        
        if  i==len(prefix)-1:
            ret = True
        else:
            ret = False
        return ret
        


# Your Trie object will be instantiated and called as such:
from pprint import pprint
trie = Trie()
print(trie.insert("apple"))
pprint(trie.trie)
print(trie.search("apple"))
print(trie.search("app"))     # 返回 false
print(trie.startsWith("app")) # 返回 true
print(trie.insert("app"))   
pprint(trie.trie)
print(trie.search("app"))     # 返回 true