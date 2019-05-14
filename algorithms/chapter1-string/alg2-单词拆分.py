#   单词拆分
# 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

# 说明：

# 拆分时可以重复使用字典中的单词。
# 你可以假设字典中没有重复的单词。
# 示例 1：

# 输入: s = "leetcode", wordDict = ["leet", "code"]
# 输出: true
# 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
# 示例 2：

# 输入: s = "applepenapple", wordDict = ["apple", "pen"]
# 输出: true
# 解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
#      注意你可以重复使用字典中的单词。
# 示例 3：

# 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
# 输出: false

class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        ret = False
        if len(s)==0:
            return ret

        wordlist = []
        i, j = 0, 0
        while(i<=j and j<=len(s)):
            if s[i:j] in wordDict:
                wordlist.append(s[i:j])
                i = j
            j = j + 1
        
        if j>len(s) and i<j:
            ret = False
        else:
            ret = True
        
        return ret

if __name__ == "__main__":
    sol = Solution()
    s = "leetcode"
    wordDict = ["leet", "code"]
    s = "applepenapple"
    wordDict = ["apple", "pen"]
    s = "catsandog"
    wordDict = ["cats", "dog", "sand", "and", "cat"]
    print(sol.wordBreak(s, wordDict))