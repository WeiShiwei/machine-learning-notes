# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = float('-inf')
        self.find_max_path_sum(root)

        return self.res

    def find_max_path_sum(self, root):
        """ 
            定义find_max_path_sum(root)为返回path的最大和路径 
            path：经过root的path
        """
        # 空节点
        if not root:
            return 0

        # 最优子结构
        max_left = self.find_max_path_sum(root.left)
        max_right = self.find_max_path_sum(root.right)
        # max_single是经过root的path对应的最优解
        # 它来作为返回值
        max_single = max(root.val, max(max_left, max_right)+root.val)
        max_root = max(max_single,
                       max_left+max_right+root.val)
        #print("debug:",root.val, max_left, max_right, max_left+max_right+root.val) 

        if self.res <= max_root:
            self.res = max_root
        
        return max_single
        