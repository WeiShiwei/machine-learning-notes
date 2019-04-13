Bidirectional RNNs 双向RNN

我们以单向的RNN举例，

在NLP任务中比如情感分析中，terribly单词对应LSTM单元的隐藏状态认为是单词的contextual表示

但是单向RNN中，terribly的contextual表示之和它的上文相关，和下文无关，这是不合逻辑的。

如何该单词的contextual表示考虑到下文呢，就需要一个反向的RNN。



![32698600-FCA4-4757-9116-001C3872C527](./images/32698600-FCA4-4757-9116-001C3872C527.png)



如下是双向RNN的结构

该结构中正向和反向的RNN没有相互影响，在输出层的每个时间步，把正向和反向的隐藏状态拼接

![D0EB82EB-A3E4-48BA-BEBC-0247C4A0FEFE](./images/D0EB82EB-A3E4-48BA-BEBC-0247C4A0FEFE.png)



Bidirectional RNNs的形式化

![D9645CF9-00EB-4A04-B623-82D70653FD4A](./images/D9645CF9-00EB-4A04-B623-82D70653FD4A.png)

如上参考CS224N课程