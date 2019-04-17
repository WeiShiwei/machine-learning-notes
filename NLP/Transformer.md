

### transfromer的定义

transfromer的网络结构由self-Attenion和Feed Forward Neural Network组成，完全由Attention机制组成；

它抛弃了传统的CNN和RNN

一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建



采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片 t的计算依赖t-1 时刻的计算结果，这样限制了模型的并行能力；
2. 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。



Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

论文中给出Transformer的定义是：Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution。



### **Self-Attention**



**Self-Attention**

编码器接收向量的list作输入。然后将其送入self-attention处理，再之后送入前向网络，最后将输入传入下一个编码器。

![img](./images/encoder_with_tensors_2.png)

每个位置的词向量被送入self-attention模块，然后是前向网络(对每个向量都是完全相同的网络结构)

**Self-Attention at a High Level**

**Self-Attention in Detail**

**Matrix Calculation of Self-Attention**

增加multi-headed的机制到self-attention



### **The Decoder Side**



### 参考

The Illustrated Transformer

<http://jalammar.github.io/illustrated-transformer/>

The Illustrated Transformer【译】

<https://blog.csdn.net/yujianmin1990/article/details/85221271>





The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)

<http://jalammar.github.io/illustrated-bert/>

迁移学习NLP：BERT、ELMo等直观图解【译】

<https://zhuanlan.zhihu.com/p/52282552>

