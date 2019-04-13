

# **Self-Attention**



**Self-Attention**

编码器接收向量的list作输入。然后将其送入self-attention处理，再之后送入前向网络，最后将输入传入下一个编码器。

![img](./images/encoder_with_tensors_2.png)

每个位置的词向量被送入self-attention模块，然后是前向网络(对每个向量都是完全相同的网络结构)

**Self-Attention at a High Level**

**Self-Attention in Detail**

**Matrix Calculation of Self-Attention**

增加multi-headed的机制到self-attention



# **The Decoder Side**



# 参考

The Illustrated Transformer

<http://jalammar.github.io/illustrated-transformer/>

The Illustrated Transformer【译】

<https://blog.csdn.net/yujianmin1990/article/details/85221271>





The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)

<http://jalammar.github.io/illustrated-bert/>

迁移学习NLP：BERT、ELMo等直观图解【译】

<https://zhuanlan.zhihu.com/p/52282552>

