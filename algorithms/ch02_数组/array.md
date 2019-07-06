# 数组的排列和组合【L2】

## 全排列

题目描述：

给定一个没有重复数字的序列，返回其所有可能的全排列。

示例:

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]



**思路分析1**

全排列首先考虑深度优先搜索，每个深度的遍历都从 0到nums.length - 1
每次循环开始需要判断当前值是否已经使用过，即 if (!tempList.contains(nums[i]))

创建一个List<Integer> tempList存放临时数据
当tempList.size() == nums.length时，res.add(new ArrayList<Integer>(tempList))将tempList存入结果
此处不能直接add.(tempList)，否则改变tempList也会导致结果的改变

特别注意每次递归结束前都需要将刚加入的值从tempList中去掉



**思路分析2**

求解关键：画图理解题意并且打印出一些信息观察程序的执行流程。

![img](./images/46-1.png)



```python
import copy

class Solution(object):
    result = []
    tmp_list = []
    
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.dfs(nums)
        return self.result
    
    def dfs(self, nums):
        if len(self.tmp_list)==len(nums):
            self.result.append(copy.copy(self.tmp_list))
            return 
        
        for i in range(0, len(nums)):
            if nums[i] not in self.tmp_list:
                self.tmp_list.append(nums[i])
                self.dfs(nums)
                self.tmp_list = self.tmp_list[:len(self.tmp_list)-1]


if __name__ == "__main__":
    # nums = [1,2,3]
    nums = [1]
    print(Solution().permute(nums))
    # [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]

```



```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums)==0:
            return list()

        ret,temp = list(),list()
        isvisted = [False]*len(nums)
        self.backtracing(ret,temp,nums,isvisted)
        return ret

    def backtracing(self, ret, temp, nums, isvisted):
        if len(temp) == len(nums):
            ret.append(temp[:])
            return

        for i in xrange(0, len(nums)):
            if not isvisted[i]:
                isvisted[i] = True
                temp.append(nums[i])
                self.backtracing(ret,temp,nums,isvisted)
                temp.pop()
                isvisted[i] = False
```



## 子集

LeetCode 78. Subsets

题目描述
给定一组不同的整数 nums，返回所有可能的子集（幂集）。

注意事项：该解决方案集不能包含重复的子集。

例如，如果 nums = [1,2,3]，结果为以下答案：

[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []



### **思路分析1**

分析
子集与全排列有两大区别

输出的List长度不等
所谓子集，就是要求该集合所包含的所有集合
所以每次循环都要将tepmList加入res
而不是等tempList.size() == nums.length
List中元素不能重复
在全排列中，结果中每个List包含的元素都相同，只是顺序不一样
如[1,2,3]和[3,2,1]
子集则不同，每个List中的元素都不相同，所以循环不能再从0开始
需要重新定义一个变量start作为dfs()的输入参数
每次递归将start设为i + 1 即不会遍历之前访问过的元素

```python
import copy

class Solution(object):
    result = []
    tmp_list = []

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.result.append([])
        nums = sorted(nums)
        self.dfs(nums, 0)
        return self.result
    
    def dfs(self, nums, start):
        if len(self.tmp_list) == len(nums):
            return 

        # 子集则不同，每个List中的元素都不相同，所以循环不能再从0开始
        # 需要重新定义一个变量start作为dfs()的输入参数
        # 每次递归将start设为i + 1 即不会遍历之前访问过的元素
        for  i in range(start, len(nums)):
            # if nums[i] not in self.tmp_list:
            self.tmp_list.append(nums[i])
            self.result.append( copy.copy(self.tmp_list) )
            self.dfs(nums, i+1)
            self.tmp_list.pop()

if __name__ == "__main__":
    nums = [1,2,3]
    print(Solution().subsets(nums))
    # [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```



### **思路分析2（更好理解！）**

更好理解！！！

下面来看递归的解法，相当于一种深度优先搜索，参见网友 [JustDoIt的博客](http://www.cnblogs.com/TenosDoIt/p/3451902.html)，由于原集合每一个数字只有两种状态，要么存在，要么不存在，那么在构造子集时就有选择和不选择两种情况，所以可以构造一棵二叉树，左子树表示选择该层处理的节点，右子树表示不选择，最终的叶节点就是所有子集合，树的结构如下：

```
                        []        
                   /          \        
                  /            \     
                 /              \
              [1]                []
           /       \           /    \
          /         \         /      \        
       [1 2]       [1]       [2]     []
      /     \     /   \     /   \    / \
  [1 2 3] [1 2] [1 3] [1] [2 3] [2] [3] []    
```



```c++
class Solution {
public:
    vector<vector<int> > subsets(vector<int> &S) {
        vector<vector<int> > res;
        vector<int> out;
        sort(S.begin(), S.end());
        getSubsets(S, 0, out, res);
        return res;
    }
    void getSubsets(vector<int> &S, int pos, vector<int> &out, vector<vector<int> > &res) {
        res.push_back(out);
        for (int i = pos; i < S.size(); ++i) {
            out.push_back(S[i]);
            getSubsets(S, i + 1, out, res);
            out.pop_back();
        }
    }
};
```



# 连续子数组的最大和

- 题目描述：

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**示例:**

```
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

**进阶:**

如果你已经实现复杂度为 O(*n*) 的解法，尝试使用更为精妙的分治法求解。



**思路分析**

动态规划的思想来解决这个问题，假定函数f(i)表示以第i个数字结尾的子数组的最大和，那么需要求出max[f(i)]

可以用如下的递归公式求f(i)

f(i) = a[i]              if i==0 or f(i-1)<=0

​        f[i-1]+a[i]     if i<>0 and f(i-1)>0

```python
def find_greatest_sum_of_subarray(a):
    if len(a)==0:
        return None
    
    greatest_sum = a[0]
    f = [0]*len(a)

    for i in range(0, len(a)):
        if i==0 or f[i-1]<=0:
            f[i] = a[i]
        else:
            f[i] = f[i-1] + a[i]
        
        if greatest_sum<f[i]:
            greatest_sum = f[i]
    
    return greatest_sum

if __name__ == "__main__":
    a = [1, -2, 3, 10, -4, 7, 2, -5]
    print(find_greatest_sum_of_subarray(a))
```

