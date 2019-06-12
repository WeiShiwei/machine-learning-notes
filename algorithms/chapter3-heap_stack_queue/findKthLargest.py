# 数组中的第K个最大元素
# 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

# 示例 1:

# 输入: [3,2,1,5,6,4] 和 k = 2
# 输出: 5
# 示例 2:

# 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
# 输出: 4
# 说明:

# 你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。


## 思路是类似快排

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        pivot = nums[-1]

        # print(k, nums)
        # partition
        left = 0
        for right in range(0, len(nums)-1):
            if nums[right]>pivot:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
        nums[-1] = nums[left]
        nums[left] = pivot

        # print(">>>", left, nums)
        which_max = left-0+1

        if which_max==k:
            return nums[left]
        elif which_max>k:
            return self.findKthLargest(nums[:left], k)
        else:
            k = k-which_max
            return self.findKthLargest(nums[left+1:], k)


if  __name__ =="__main__":
    nums = [3,2,1,5,6,4]
    k = 2

    nums = [3,2,3,1,2,4,5,5,6]
    k = 4

    print(Solution().findKthLargest(nums, k))