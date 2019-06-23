# 1. 二叉树的遍历

## 1.1 递归遍历

## 1.2 非递归遍历

### 1.2.1 前序非递归

### 1.2.2 中序非递归

### 1.2.3 后续非递归

## 1.3 层次遍历



# 2. 树的面试题

## 2.1 二叉搜索树中第K小的元素

给定一个二叉搜索树，编写一个函数 `kthSmallest` 来查找其中第 **k** 个最小的元素。

**说明：**
你可以假设 k 总是有效的，1 ≤ k ≤ 二叉搜索树元素个数。

**示例 1:**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 1
```

**示例 2:**

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 3
```

**进阶：**
如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化 `kthSmallest` 函数？



【方法一】

1、计算左子树元素个数left_treesize。

2、 left_treesize+1 = K，则根节点即为第K个元素

​    left_treesize >=k, 则第K个元素在左子树中，

​    left_treesize +1 <k, 则转换为在右子树中，寻找第K-left_treesize-1元素。

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def cal_treesize(self, root):
        if root is None:
            return 0
        return 1+self.cal_treesize(root.left)+self.cal_treesize(root.right)
    
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        
        left_treesize = self.cal_treesize(root.left)
        if left_treesize == k-1:
            return root.val
        
        if left_treesize < k-1:
            return self.kthSmallest(root.right, k-left_treesize-1)
        
        if left_treesize > k-1:
            return self.kthSmallest(root.left, k)

```



【方法二】

 因中序遍历为一个有序的数组，所以可以在中序遍历的过程中进行比较

```c++
int kthSmallest(TreeNode root, int k) {  
        Stack<TreeNode *> stack ;  
        if(root == NULL || K<=0)
              return -1;//表示不存在
        
        TreeNode *P = root;  
        while(P->left != NULL) {  //也是个循环
            stack.push(P);  
            P = P->left;   
        }  
          
        while(k>0 && (P != NULL || !stack.empty())) {  //注意k这个条件别忘记
            if(p==NULL) {  
                p = stack.top();
                stack.pop();
                //查找输出的过程中进行判断，是否为第k个元素
                 if(--k==0) 
                     return p->val;  
                
                  p = p->right;  
            } else {  
                stack.push(p);  
                p = p->left;  
            }  
        }   
    }  
```

  

## 2.2 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]

![img](./images/binarytree.png)

 

**示例 1:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```

**示例 2:**

```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```



**解题思路：**

递归搜索左右子树，如果左子树和右子树都不为空，说明最近父节点一定在根节点。

反之，如果左子树为空，说明两个节点一定在右子树；

同理如果右子树为空，说明两个节点一定在左子树。



```python
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None: return root
        if root==p or root==q: return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if (left is not None) and (right is not None):
            return root
        
        if left is None:
            return right
        
        if right is None:
            return left
```



## 2.3 二叉树的序列化与反序列化

序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

**示例:** 

```
你可以将以下二叉树：

    1
   / \
  2   3
     / \
    4   5

序列化为 "[1,2,3,null,null,4,5]"
```

**提示:** 这与 LeetCode 目前使用的方式一致，详情请参阅 [LeetCode 序列化二叉树的格式](https://leetcode-cn.com/faq/#binary-tree)。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

**说明:** 不要使用类的成员 / 全局 / 静态变量来存储状态，你的序列化和反序列化算法应该是无状态的。



```python
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:    
    def serialize(self, root):
        """Encodes a tree to a single string.

        前序遍历：
            1
            / \
            2   3
                / \
                4   5
            前序遍历: 12$$34$$5$$
        
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return '$'
        return ','.join([ str(root.val), str(self.serialize(root.left)), str(self.serialize(root.right)) ])

    def deserialize_core(self, list):
        """
            Paras:
                p是TreeNode的引用
                data='12$$34$$5$$'
        """
        if len(list)==0:
            return None

        ch = list.pop(0)
        root = None
        if ch != '$':
            root = TreeNode(int(ch))
            root.left = self.deserialize_core(list)
            root.right = self.deserialize_core(list)
        return root


    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        list = data.split(',')
        root = self.deserialize_core(list)
        return root

if __name__ == "__main__":
    data = '1,2,$,$,3,4,$,$,5,$,$'
    codec = Codec()
    root = codec.deserialize(data)
    print(codec.serialize(root))
    # 1,2,$,$,3,4,$,$,5,$,$
```

注意：

1、参考剑指offer面试题62：序列化二叉树

因为语言的特点，python和C++的代码实现上区别还是挺大的



## 2.4 根据一棵树的前序遍历与中序遍历构造二叉树

leetcode105. 从前序与中序遍历序列构造二叉树

根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
返回如下的二叉树：

    		3
       / \
      9  20
        /  \
       15   7
```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # 边界处理
        if len(preorder)<=0 or len(inorder)<=0:
            return None
        
        # 前序遍历的第一个值是根节点
        val = preorder[0]
        root = TreeNode(val)

        # 在中序遍历中找到根节点值对应的索引index
        index = -1
        for i, elm in enumerate(inorder):
            if elm==val:
                index = i
                break
        # 异常处理
        if index==-1:
            raise ValueError()

        left_size = index
        right_size = len(inorder)-left_size-1
        if left_size > 0: # 构建左子树
            root.left = self.buildTree(preorder[1:1+left_size], inorder[:left_size])
        if right_size > 0: # 构建右子树
            root.right = self.buildTree(preorder[-right_size:], inorder[-right_size:])

        return root

if __name__ == "__main__":
    preorder = [3,9,20,15,7]
    inorder = [9,3,15,20,7]
    Solution().buildTree(preorder, inorder)
```


