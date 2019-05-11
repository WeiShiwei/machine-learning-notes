

**Vanishing gradient problem** 

**Long Short-Term Memory (LSTM)**

**Gated Recurrent Units (GRU)**

**Is vanishing/exploding gradient just a RNN problem?**

**Bidirectional RNNs**

**Multi-layer RNNs**



#  LSTM

Long Short Term Memory

超生动图解LSTM和GRU：拯救循环神经网络的记忆障碍，就靠它们了

https://zhuanlan.zhihu.com/p/46981722



# GRU

![img](./images/v2-838b5aeac6a58f3c478fbae62c049f84_b.gif)

![img](https://pic1.zhimg.com/v2-9701d76234ace3f429bd566bf47b10bc_b.gif)



# Bidirectional RNNs
我们以单向的RNN举例，在NLP任务中比如情感分析中，terribly单词对应LSTM单元的隐藏状态认为是单词的contextual表示，但是单向RNN中，terribly的contextual表示只和它的上文相关，和下文无关，这是不合逻辑的。

如何该单词的contextual表示考虑到下文呢，就需要一个反向的RNN。



![32698600-FCA4-4757-9116-001C3872C527](./images/32698600-FCA4-4757-9116-001C3872C527.png)





如下是双向RNN的结构

该结构中正向和反向的RNN没有相互影响，在输出层的每个时间步，把正向和反向的隐藏状态拼接

![D0EB82EB-A3E4-48BA-BEBC-0247C4A0FEFE](./images/D0EB82EB-A3E4-48BA-BEBC-0247C4A0FEFE.png)



Bidirectional RNNs的形式化

![D9645CF9-00EB-4A04-B623-82D70653FD4A](./images/D9645CF9-00EB-4A04-B623-82D70653FD4A.png)





# Multi-layer RNNs

NEXT




# 参考

1、超生动图解LSTM和GRU：拯救循环神经网络的记忆障碍，就靠它们了

https://zhuanlan.zhihu.com/p/46981722





