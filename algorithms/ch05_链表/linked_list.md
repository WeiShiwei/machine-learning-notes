# 1. 环形链表

给定一个链表，判断链表中是否有环。

为了表示给定链表中的环，我们使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 `pos` 是 `-1`，则在该链表中没有环。

 

**示例 1：**

```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
```

![img](./images/circularlinkedlist.png)

**示例 2：**

```
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
```

![img](./images/circularlinkedlist_test2.png)

**示例 3：**

```
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
```

![img](./images/circularlinkedlist_test3.png)

 

**进阶：**

你能用 *O(1)*（即，常量）内存解决此问题吗？

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        header1 = head
        header2 = head

        while((header1 is not None) and (header2 is not None)):
            header1 = header1.next
            if header2.next:
                header2 = header2.next.next
            else:
                break
            
            if header1 == header2:
                return True
        
        return False

```



# 2. 排序链表

在 *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

**示例 1:**

```
输入: 4->2->1->3
输出: 1->2->3->4
```

**示例 2:**

```
输入: -1->5->3->4->0
输出: -1->0->3->4->5
```



思路分析：

对一个链表进行排序，且时间复杂度要求为 O(n log n) ，空间复杂度为常量。一看到 O(n log n) 的排序，首先应该想到归并排序和快速排序，但是通常我们使用这两种排序方法时都是针对数组的，现在是链表了。

​        归并排序法：在动手之前一直觉得空间复杂度为常量不太可能，因为原来使用归并时，都是 O(N)的，需要复制出相等的空间来进行赋值归并。对于链表，实际上是可以实现常数空间占用的（链表的归并排序不需要额外的空间）。利用归并的思想，递归地将当前链表分为两段，然后merge，分两段的方法是使用 fast-slow 法，用两个指针，一个每次走两步，一个走一步，知道快的走到了末尾，然后慢的所在位置就是中间位置，这样就分成了两段。merge时，把两段头部节点值比较，用一个 p 指向较小的，且记录第一个节点，然后 两段的头一步一步向后走，p也一直向后走，总是指向较小节点，直至其中一个头为NULL，处理剩下的元素。最后返回记录的头即可。

主要考察3个知识点，
知识点1：归并排序的整体思想
知识点2：找到一个链表的中间节点的方法
知识点3：合并两个已排好序的链表为一个新的有序链表

归并排序的基本思想是：找到链表的middle节点，然后递归对前半部分和后半部分分别进行归并排序，最后对两个以排好序的链表进行Merge。



```python
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        return self.merge_sort(head)
        
    def merge_sort(self, head):
        # 边界条件
        # 如果head指向链表size小于2，直接return
        if head.next is None:
            return head

        # 链表分成left和right
        # fast-slow 法，用两个指针，一个每次走两步，一个走一步，
        # 知道快的走到了末尾，然后慢的所在位置就是中间位置，这样就分成了两段
        p1 = p2 = head
        prev = None
        while(p1 and p2):
            prev = p1
            p1 = p1.next
            if p2.next:
                p2 = p2.next.next
            else:
                break
        
        prev.next = None
        
        left = self.merge_sort(head)
        right = self.merge_sort(p1)
        
        # 合并left和right
        return self.merge(left, right)
    
    def merge(self, left, right):
        # header指向空节点
        # p设置为header
        header = ListNode(-1)
        p = header

        while(left and right):
            if left.val <= right.val:
                p.next = left
                p = p.next
                left = left.next
            else:
                p.next = right
                p = p.next
                right = right.next
        
        if left:
            p.next = left
        
        if right:
            p.next = right

        p = header.next
        del header
        return p

```

