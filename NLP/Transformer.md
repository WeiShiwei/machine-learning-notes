

### transfromer的定义

transfromer的网络结构由self-Attenion和Feed Forward Neural Network组成，完全由Attention机制组成；

它抛弃了传统的CNN和RNN

一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建



采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片 t的计算依赖t-1 时刻的计算结果，这样限制了模型的并行能力；
2. 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。



Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

论文中给出Transformer的定义是：Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution。



### Attention

Attention model和seq2seq模型紧密联系的

Attention解决了基于encoder-decoder结构下seq2seq模型的信息瓶颈问题。

Attention的核心思想是在decoder的每个时间步，选择与encoder直接连接的方式，专注于source序列相应的部分。



在seq2seq模型应用attention如下图所示：

1、在decoder里面每个单元的输出前，计算该单元hidden status与encoder各个单元的hidden status计算点积，获得attention分数

2、然后softmax归一化处理获得attention分布

3、attention分布作为权重对encoder中各个单元hidden status求加权和

4、把decoder该单元的attention输出与hidden status，计算输出



优点是：

-  Attention significantly improves NMT performance

It’s very useful to allow decoder to focus on certain parts of the source

-  Attention solves the bottleneck problem

Attention allows decoder to look directly at source; bypass bottleneck

- Attention helps with vanishing gradient problem 
- Attention provides some interpretability 

但是缺点是：

1、Attention不能并行化；

2、Attention忽略了输入句子、目标句子之间的关系；



![attention_1555669497846](./images/attention_1555669497846.jpg)





![attention_1555668254026](./images/attention_1555668254026.jpg)



### **Self-Attention**

#### **Self-Attention at a High Level**

#### **Self-Attention in Detail**

#### multi-headed机制的self-attention



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

