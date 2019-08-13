# 基于注意力机制，机器之心带你理解与训练神经机器翻译系统



本文是机器之心 GitHub 实现项目，我们根据谷歌的 Transformer 原论文与 Harvard NLP 所实现的代码学习构建了一个神经机器翻译系统。因此，我们希望各位读者也能根据这篇文章了解 Transformer 的架构，并动手实现一个神经机器翻译系统。

自去年 6 月份「Attention is All You Need」发表以来，Transformer 受到越来越多的关注。它除了能显著提升翻译质量，同时还为很多 NLP 任务提供了新的架构。这篇论文放弃了传统基于 RNN 或 CNN 的深度架构，并只保留了注意力（Attentaion）机制，虽然原论文在这一方面描述地比较清楚，但要正确地实现这样的新型架构可能非常困难。

在这篇文章中，我们从注意力机制到神经机器翻译系统解释了实现 Transformer 的架构与代码，并借助这些实现理解原论文。机器之心整理了整个实现，并根据我们对原论文与实现的理解添加一些解释。整个文章就是一个可运行的 Jupyter Notebook，读者可直接在 Colaboratory 中阅读文章与运行代码。

- 机器之心实现地址：https://github.com/jiqizhixin/ML-Tutorial-Experiment 
- 原实现地址：https://github.com/harvardnlp/annotated-transformer

本文所有的代码都可以在谷歌 Colab 上运行，且读者也可以在 GitHub 中下载全部的代码在本地运行。这篇文章非常适合于研究者与感兴趣的开发者，代码很大程度上都依赖于 OpenNMT 库。

在运行模型前，我们需要确保有对应的环境。如果在本地运行，那么需要确保以下基本库的导入不会报错，若在 Colab 上运行，那么首先需要运行以下第一个 pip 语句安装对应的包。Colab 的环境配置非常简单，一般只需要使用 conda 或 pip 命令就能完成。此外，Colab 语句前面加上「!」表示这是命令行，而不加感叹号则表示这个代码框是 Python 代码。

```
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
```

**引言**

减少序列计算的任务目标构成了 Extended Neural GPU、ByteNet 和 ConvS2S 的基础，它们都是使用卷积神经网络作为基本构建块，因而能对所有输入与输出位置的隐藏表征执行并行计算。在这些模型中，两个任意输入与输出位置的信号关联所需要的运算数量与它们的位置距离成正比，对于 ConvS2S 为线性增长，对于 ByteNet 为对数增长。这种现象使得学习较远位置的依赖关系非常困难。而在 Transformer 中，这种成本会减少到一个固定的运算数量，尽管平均注意力位置加权会减少有效表征力，但使用 Multi-Head Attention 注意力机制可以抵消这种成本。

自注意力（Self-attention），有时也称为内部注意力，它是一种涉及单序列不同位置的注意力机制，并能计算序列的表征。自注意力在多种任务中都有非常成功的应用，例如阅读理解、摘要概括、文字蕴含和语句表征等。自注意力这种在序列内部执行 Attention 的方法可以视为搜索序列内部的隐藏关系，这种内部关系对于翻译以及序列任务的性能非常重要。

然而就我们所知道的，Transformer 是第一种完全依赖于自注意力以计算输入与输出表征的方法，这意味着它没有使用序列对齐的 RNN 或卷积网络。从 Transformer 的结构就可以看出，它并没有使用深度网络抽取序列特征，顶多使用几个线性变换对特征进行变换。

本文主要从模型架构、训练配置和两个实际翻译模型开始介绍 Ashish Vaswani 等人的原论文与 Harvard NLP 团队实现的代码。在模型架构中，我们将讨论编码器、解码器、注意力机制以及位置编码等关键组成部分，而训练配置将讨论如何抽取批量数据、设定训练循环、选择最优化方法和正则化器等。最后我们将跟随 Alexander Rush 等人的实现训练两个神经机器翻译系统，其中一个仅使用简单的合成数据，而另一个则是真实的 IWSLT 德语-英语翻译数据集。

**模型架构**

大多数神经序列模型都使用编码器-解码器框架，其中编码器将表征符号的输入序列 (x_1, …, x_n) 映射到连续表征 z=(z_1, …, z_n)。给定中间变量 z，解码器将会生成一个输出序列 (y_1,…,y_m)。在每一个时间步上，模型都是自回归的（auto-regressive），当生成序列中的下一个元素时，先前生成的元素会作为输入。

以下展示了一个标准的编码器-解码器框架，EncoderDecoder 类定义了先编码后解码的过程，例如先将英文序列编码为一个隐向量，在基于这个中间表征解码为中文序列。

```
class EncoderDecoder(nn.Module):
 """
 A standard Encoder-Decoder architecture. Base for this and many 
 other models.
 """
 def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
 super(EncoderDecoder, self).__init__()
 self.encoder = encoder
 self.decoder = decoder
 self.src_embed = src_embed
 self.tgt_embed = tgt_embed
 self.generator = generator

 def forward(self, src, tgt, src_mask, tgt_mask):
 "Take in and process masked src and target sequences."
 return self.decode(self.encode(src, src_mask), src_mask,
 tgt, tgt_mask)

 def encode(self, src, src_mask):
 return self.encoder(self.src_embed(src), src_mask)

 def decode(self, memory, src_mask, tgt, tgt_mask):
 return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
class Generator(nn.Module):
 "Define standard linear + softmax generation step."
 def __init__(self, d_model, vocab):
 super(Generator, self).__init__()
 self.proj = nn.Linear(d_model, vocab)

 def forward(self, x):
 return F.log_softmax(self.proj(x), dim=-1)
```

Transformer 的整体架构也采用了这种编码器-解码器的框架，它使用了多层自注意力机制和层级归一化，编码器和解码器都会使用全连接层和残差连接。Transformer 的整体结构如下图所示：

![img](https://image.jiqizhixin.com/uploads/editor/1da580c8-fd5d-468f-b440-2eeaba95f06c/1526094613305.png)

如上所示，左侧为输入序列的编码器。输入序列首先会转换为词嵌入向量，在与位置编码向量相加后可作为 Multi-Head Attention 模块的输入，该模块的输出在与输入相加后将投入层级归一化函数，得出的输出在馈送到全连接层后可得出编码器模块的输出。这样相同的 6 个编码器模块（N=6）可构成整个编码器架构。解码器模块首先同样构建了一个自注意力模块，然后再结合编码器的输出实现 Multi-Head Attention，最后投入全连接网络并输出预测词概率。

这里只是简单地介绍了模型的大概过程，很多如位置编码、Multi-Head Attention 模块、层级归一化、残差链接和逐位置前馈网络等概念都需要读者详细阅读下文，最后再回过头理解完整的过程。

**编码器与解码器堆栈**

- 编码器

编码器由相同的 6 个模块堆叠而成，每一个模块都有两个子层级构成。其中第一个子层级是 Multi-Head 自注意机制，其中自注意力表示输入和输出序列都是同一条。第二个子层级采用了全连接网络，主要作用在于注意子层级的特征。此外，每一个子层级都会添加一个残差连接和层级归一化。

以下定义了编码器的主体框架，在 Encoder 类中，每一个 layer 表示一个编码器模块，这个编码器模块由两个子层级组成。layer 函数的输出表示经过层级归一化的编码器模块输出，通过 For 循环堆叠层级就能完成整个编码器的构建。

```
def clones(module, N):
 "Produce N identical layers."
 return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
 "Core encoder is a stack of N layers"
 def __init__(self, layer, N):
 super(Encoder, self).__init__()
 self.layers = clones(layer, N)
 self.norm = LayerNorm(layer.size)

 def forward(self, x, mask):
 "Pass the input (and mask) through each layer in turn."
 for layer in self.layers:
 x = layer(x, mask)
 return self.norm(x)
```

如编码器的结构图所示，每个子层级都会会添加一个残差连接，并随后传入层级归一化。上面构建的主体架构也调用了层级归一化函数，以下代码展示了层级归一化的定义。

```
class LayerNorm(nn.Module):
 "Construct a layernorm module (See citation for details)."
 def __init__(self, features, eps=1e-6):
 super(LayerNorm, self).__init__()
 self.a_2 = nn.Parameter(torch.ones(features))
 self.b_2 = nn.Parameter(torch.zeros(features))
 self.eps = eps

 def forward(self, x):
 mean = x.mean(-1, keepdim=True)
 std = x.std(-1, keepdim=True)
 return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

层级归一化可以通过修正每一层内激活值的均值与方差而大大减少协方差偏离问题。简单来说，一个层级的均值可以通过计算该层所有神经元激活值的平均值而得出，然后再根据均值计算该层所有神经元激活值的方差。最后根据均值与方差，我们可以对这一层所有输出值进行归一化。

如上 LayerNorm 类所示，我们首先需要使用方法 mean 求输入 x 最后一个维度的均值，keepdim 为真表示求均值后的维度保持不变，并且均值会广播操作到对应的维度。同样使用 std 方法计算标准差后，该层所有激活值分别减去均值再除以标准差就能实现归一化，分母加上一个小值 eps 可以防止分母为零。

因此，每一个子层的输出为 LayerNorm(x+Sublayer(x))，其中 Sublayer(x) 表示由子层本身实现的函数。我们应用 Dropout 将每一个子层的输出随机失活，这一过程会在加上子层输入和执行归一化之前完成。

以下定义了残差连接，我们会在投入层级归一化函数前将子层级的输入与输出相加。为了使用这些残差连接，模型中所有的子层和嵌入层的输出维度都是 d_model=512。

```
class SublayerConnection(nn.Module):
 """
 A residual connection followed by a layer norm.
 Note for code simplicity the norm is first as opposed to last.
 """
 def __init__(self, size, dropout):
 super(SublayerConnection, self).__init__()
 self.norm = LayerNorm(size)
 self.dropout = nn.Dropout(dropout)

 def forward(self, x, sublayer):
 "Apply residual connection to any sublayer with the same size."
 return x + self.dropout(sublayer(self.norm(x)))
```

在上述代码定义中，x 表示上一层添加了残差连接的输出，这一层添加了残差连接的输出需要将 x 执行层级归一化，然后馈送到 Multi-Head Attention 层或全连接层，添加 Dropout 操作后可作为这一子层级的输出。最后将该子层的输出向量与输入向量相加得到下一层的输入。

编码器每个模块有两个子层，第一个为 multi-head 自注意力层，第二个为简单的逐位置全连接前馈网络。以下的 EncoderLayer 类定义了一个编码器模块的过程。

```
class EncoderLayer(nn.Module):
 "Encoder is made up of self-attn and feed forward (defined below)"
 def __init__(self, size, self_attn, feed_forward, dropout):
 super(EncoderLayer, self).__init__()
 self.self_attn = self_attn
 self.feed_forward = feed_forward
 self.sublayer = clones(SublayerConnection(size, dropout), 2)
 self.size = size

 def forward(self, x, mask):
 "Follow Figure 1 (left) for connections."
 x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
 return self.sublayer[1](x, self.feed_forward)
```

以上代码叠加了自注意力层与全连接层，其中 Multi-Head Attention 机制的输入 Query、Key 和 Value 都为 x 就表示自注意力。

- 解码器

解码器也由相同的 6 个模块堆叠而成，每一个解码器模块都有三个子层组成，每一个子层同样会加上残差连接与层级归一化运算。第一个和第三个子层分别与编码器的 Multi-Head 自注意力层和全连接层相同，而第二个子层所采用的 Multi-Head Attention 机制使用编码器的输出作为 Key 和 Value，采用解码模块第一个子层的输出作为 Query。

我们同样需要修正编码器堆栈中的自注意力子层，以防止当前位置注意到后续序列位置，这一修正可通过掩码实现。以下的解码器的主体堆叠结构和编码器相似，只需要简单地堆叠解码器模块就能完成。

```
class Decoder(nn.Module):
 "Generic N layer decoder with masking."
 def __init__(self, layer, N):
 super(Decoder, self).__init__()
 self.layers = clones(layer, N)
 self.norm = LayerNorm(layer.size)

 def forward(self, x, memory, src_mask, tgt_mask):
 for layer in self.layers:
 x = layer(x, memory, src_mask, tgt_mask)
 return self.norm(x)
```

以下展示了一个解码器模块的架构，第一个 Multi-Head Attention 机制的三个输入都是 x，因此它是自注意力。第二个 Multi-Head 注意力机制输入的 Key 和 Value 是编码器的输出 memory，输入的 Query 是上一个子层的输出 x。最后在叠加一个全连接网络以完成一个编码器模块的构建。

```
class DecoderLayer(nn.Module):
 "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
 def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
 super(DecoderLayer, self).__init__()
 self.size = size
 self.self_attn = self_attn
 self.src_attn = src_attn
 self.feed_forward = feed_forward
 self.sublayer = clones(SublayerConnection(size, dropout), 3)

 def forward(self, x, memory, src_mask, tgt_mask):
 "Follow Figure 1 (right) for connections."
 m = memory
 x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
 x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
 return self.sublayer[2](x, self.feed_forward)
```

对于序列建模来说，模型应该只能查看有限的序列信息。例如在时间步 i，模型能读取整个输入序列，但只能查看时间步 i 及之前的序列信息。对于 Transformer 的解码器来说，它会输入整个目标序列，且注意力机制会注意到整个目标序列各个位置的信息，因此我们需要限制注意力机制能看到的信息。

如上所述，Transformer 在注意力机制中使用 subsequent_mask 函数以避免当前位置注意到后面位置的信息。因为输出词嵌入是位置的一个偏移，因此我们可以确保位置 i 的预测仅取决于在位置 i 之前的已知输出。

```
def subsequent_mask(size):
 "Mask out subsequent positions."
 attn_shape = (1, size, size)
 subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
 return torch.from_numpy(subsequent_mask) == 0
```

以下为注意力掩码的可视化，其中每一行为一个词，每一列则表示一个位置。下图展示了每一个词允许查看的位置，训练中词是不能注意到未来词的。

```
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
```

![img](https://image.jiqizhixin.com/uploads/editor/0fa18a6f-6b3c-4b47-bb79-033e2de4ebc3/1526094612561.png)

**注意力机制**

谷歌在原论文中展示了注意力机制的一般化定义，即它和 RNN 或 CNN 一样也是一种编码序列的方案。一个注意力函数可以描述为将 Query 与一组键值对（Key-Value）映射到输出，其中 Query、Key、Value 和输出都是向量。输出可以通过值的加权和而计算得出，其中分配到每一个值的权重可通过 Query 和对应 Key 的适应度函数（compatibility function）计算。

在翻译任务中，Query 可以视为原语词向量序列，而 Key 和 Value 可以视为目标语词向量序列。一般的注意力机制可解释为计算 Query 和 Key 之间的相似性，并利用这种相似性确定 Query 和 Value 之间的注意力关系。

以下是点积注意力的结构示意图，我们称这种特殊的结构为「缩放点积注意力」。它的输入由维度是 d_k 的 Query 和 Key 组成，Value 的维度是 d_v。如下所示，我们会先计算 Query 和所有 Key 的点乘，并每一个都除上 squre_root(d_k) 以防止乘积结果过大，然后再馈送到 Softmax 函数以获得与 Value 对应的权重。根据这样的权重，我们就可以配置 Value 向量而得出最后的输出。

```
Image(filename='images/ModalNet-19.png')
```

![img](https://image.jiqizhixin.com/uploads/editor/f8f054c7-9416-44de-be13-54c0ded379e1/1526094613045.png)

在上图中，Q 和 K 的运算有一个可选的 Mask 过程。在编码器中，我们不需要使用它限制注意力模块所关注的序列信息。而在解码器中，我们需要它限制注意力模块只能注意到当前时间步及之前时间步的信息。这一个过程可以很简洁地表示为函数 Attention(Q, K, V)。

Attention(Q, K, V) 函数在输入矩阵 Q、K 和 V 的情况下可计算 Query 序列与 Value 序列之间的注意力关系。其中 Q 的维度为 n×d_k，表示有 n 条维度为 d_k 的 Query、K 的维度为 m×d_k、V 的维度为 m×d_v。这三个矩阵的乘积可得出 n×d_v 维的矩阵，它表示 n 条 Query 对应注意到的 Value 向量。

![img](https://image.jiqizhixin.com/uploads/editor/68e49e43-1535-42d8-80c7-82f1a82239a8/1526094613179.png)

上式中 Q 与 K 的点积会除上 squre_root(d_k) 以实现缩放。原论文作者发现，当每一条 Query 的维度 d_k 比较小时，点乘注意力和加性注意力的性能相似，但随着 d_k 的增大，加性注意力的性能会超过点乘注意力机制。不过点乘注意力有一个强大的属性，即它可以利用矩阵乘法的并行运算大大加快训练速度。

原论文作者认为点乘注意力效果不好的原因是在 d_k 比较大的情况下，乘积结果会非常大，因此会导致 Softmax 快速饱和并只能提供非常小的梯度来更新参数。所以他们采用了根号下 d_k 来缩小点乘结果，并防止 Softmax 函数饱和。

为了证明为什么点积的量级会变得很大，我们假设元素 q 和 k 都是均值为 0、方差为 1 的独立随机变量，它们的点乘 q⋅k=∑q_i*k_i 有 0 均值和 d_k 的方差。为了抵消这种影响，我们可以通过除上 squre_root(d_k) 以归一化点乘结果。

以下函数定义了一个标准的点乘注意力，该函数最终会返回匹配 Query 和 Key 的权重或概率 p_attn，以及最终注意力机制的输出序列。

```
def attention(query, key, value, mask=None, dropout=None):
 "Compute 'Scaled Dot Product Attention'"
 d_k = query.size(-1)
 scores = torch.matmul(query, key.transpose(-2, -1)) \
 / math.sqrt(d_k)
 if mask is not None:
 scores = scores.masked_fill(mask == 0, -1e9)
 p_attn = F.softmax(scores, dim = -1)
 if dropout is not None:
 p_attn = dropout(p_attn)
 return torch.matmul(p_attn, value), p_attn
```

在上述函数中，query 矩阵的列数即维度数 d_k。在计算点乘并缩放后，我们可以在最后一个维度执行 Softmax 函数以得到概率 p_attn。

两个最常见的注意力函数是加性注意力（additive attention）和点乘（乘法）注意力。除了要除上缩放因子 squre_root(d_k)，标准的点乘注意力与原论文中所采用的是相同的。加性注意力会使用单隐藏层的前馈网络计算适应度函数，它们在理论复杂度上是相似的。点积注意力在实践中更快速且参数空间更高效，因为它能通过高度优化的矩阵乘法库并行地计算。

**Multi-head Attention** 

下图展示了 Transformer 中所采用的 Multi-head Attention 结构，它其实就是多个点乘注意力并行地处理并最后将结果拼接在一起。一般而言，我们可以对三个输入矩阵 Q、V、K 分别进行 h 个不同的线性变换，然后分别将它们投入 h 个点乘注意力函数并拼接所有的输出结果。

```
Image(filename='images/ModalNet-20.png')
```

![img](https://image.jiqizhixin.com/uploads/editor/c826c126-4dac-4d18-87d3-a621d503977b/1526094612960.png)

Multi-head Attention 允许模型联合关注不同位置的不同表征子空间信息，我们可以理解为在参数不共享的情况下，多次执行点乘注意力。Multi-head Attention 的表达如下所示：

![img](https://image.jiqizhixin.com/uploads/editor/72e32153-52df-479a-bc76-c3b633e3c4ef/1526094612476.png)

其中 W 为对应线性变换的权重矩阵，Attention() 就是上文所实现的点乘注意力函数。

在原论文和实现中，研究者使用了 h=8 个并行点乘注意力层而完成 Multi-head Attention。对于每一个注意力层，原论文使用的维度是 d_k=d_v=d_model/h=64。由于每一个并行注意力层的维度降低，总的计算成本和单个点乘注意力在全维度上的成本非常相近。

以下定义了 Multi-head Attention 模块，它实现了上图所示的结构：

```
class MultiHeadedAttention(nn.Module):
 def __init__(self, h, d_model, dropout=0.1):
 "Take in model size and number of heads."
 super(MultiHeadedAttention, self).__init__()
 assert d_model % h == 0
 # We assume d_v always equals d_k
 self.d_k = d_model // h
 self.h = h
 self.linears = clones(nn.Linear(d_model, d_model), 4)
 self.attn = None
 self.dropout = nn.Dropout(p=dropout)

 def forward(self, query, key, value, mask=None):
 "Implements Figure 2"
 if mask is not None:
 # Same mask applied to all h heads.
 mask = mask.unsqueeze(1)
 nbatches = query.size(0)

 # 1) Do all the linear projections in batch from d_model => h x d_k 
 query, key, value = \
 [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
 for l, x in zip(self.linears, (query, key, value))]

 # 2) Apply attention on all the projected vectors in batch. 
 x, self.attn = attention(query, key, value, mask=mask, 
 dropout=self.dropout)

 # 3) "Concat" using a view and apply a final linear. 
 x = x.transpose(1, 2).contiguous() \
 .view(nbatches, -1, self.h * self.d_k)
 return self.linears[-1](x)
```

在以上代码中，首先我们会取 query 的第一个维度作为批量样本数，然后再实现多个线性变换将 d_model 维的词嵌入向量压缩到 d_k 维的隐藏向量，变换后的矩阵将作为点乘注意力的输入。点乘注意力输出的矩阵将在最后一个维度拼接，即 8 个 n×64 维的矩阵拼接为 n×512 维的大矩阵，其中 n 为批量数。这样我们就将输出向量恢复为与词嵌入向量相等的维度。

前面我们已经了解到 Transformer 使用了大量的自注意力机制，即 Attention(X, X, X )。简单而言，Transformer 使用自注意力代替 RNN 或 CNN 抽取序列特征。对于机器翻译任务而言，自注意力输入的 Query、Key 和 Value 都是相同的矩阵，那么 Query和 Key 之间的运算就相当于计算输入序列内部的相似性，并根据这种相似性或权重注意到序列自身（Value）的内部联系。

这种内部联系可能是主语注意到谓语和宾语的信息或其它隐藏在句子内部的结构。Transformer 在神经机器翻译和阅读理解等任务上的优秀性能，都证明序列内部结构的重要性。

Transformer 以三种不同的方式使用 multi-head Attention。首先在编码器到解码器的层级中，Query 来源于前面解码器的输出，而记忆的 Key 与 Value 都来自编码器的输出。这允许解码器中的每一个位置都注意输入序列中的所有位置，因此它实际上模仿了序列到序列模型中典型的编码器-解码器注意力机制。

其次，编码器包含了自注意力层，且该层中的所有 Value、Key 和 Query 都是相同的输入矩阵，即编码器的前层输出。最后，解码器中的自注意力层允许解码器中的每一个位置都注意到包括当前位置的所有合法位置。这可以通过上文定义的 Mask 函数实现，从而防止产生左向信息流来保持自回归属性。

**逐位置的前馈网络**

为了注意子层，每一个编码器和解码器模块最后都包含一个全连接前馈网络，它独立且相同地应用于每一个位置。这个前馈网络包含两个线性变换和一个非线性激活函数，且在训练过程中我们可以在两层网络之间添加 Dropout 方法：

![img](https://image.jiqizhixin.com/uploads/editor/5f80190c-f72a-4669-a17b-e59041794f96/1526094613215.png)

如果我们将这两个全连接层级与残差连接和层级归一化结合，那么它就是每一个编码器与解码器模块最后所必须的子层。我们可以将这一子层表示为：LayerNorm(x + max(0, x*w1 + b1)w2 + b2)。

尽管线性变换在所有不同的位置上都相同，但在不同的层级中使用不同的参数，这种变换其实同样可以描述为核大小为 1 的两个卷积。输入和输出的维度 d_model=512，而内部层级的维度 d_ff=2018。

如下所示，前馈网络的定义和常规的方法并没有什么区别，不过这个网络没有添加偏置项，且对第一个全连接的输出实现了 Dropout 以防止过拟合。

```
class PositionwiseFeedForward(nn.Module):
 "Implements FFN equation."
 def __init__(self, d_model, d_ff, dropout=0.1):
 super(PositionwiseFeedForward, self).__init__()
 self.w_1 = nn.Linear(d_model, d_ff)
 self.w_2 = nn.Linear(d_ff, d_model)
 self.dropout = nn.Dropout(dropout)

 def forward(self, x):
 return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

**词嵌入和 Softmax**

与其它序列模型相似，我们可以使用学得的词嵌入将输入和输出的词汇转换为维度等于 d_model 的向量。我们还可以使用一般的线性变换和 Softmax 函数将解码器的输出转化为预测下一个词汇的概率。在愿论文的模型中，两个嵌入层和 pre-softmax 线性变换的权重矩阵是共享的。在词嵌入层中，我们将所有权重都乘以 squre_root(d_model)。

```
class Embeddings(nn.Module):
 def __init__(self, d_model, vocab):
 super(Embeddings, self).__init__()
 self.lut = nn.Embedding(vocab, d_model)
 self.d_model = d_model

 def forward(self, x):
 return self.lut(x) * math.sqrt(self.d_model)
```

**位置编码**

位置编码是 Transformer 模型中最后一个需要注意的结构，它对使用注意力机制实现序列任务也是非常重要的部分。如上文所述，Transformer 使用自注意力机制抽取序列的内部特征，但这种代替 RNN 或 CNN 抽取特征的方法有很大的局限性，即它不能捕捉序列的顺序。这样的模型即使能根据语境翻译出每一个词的意义，那也组不成完整的语句。

为了令模型能利用序列的顺序信息，我们必须植入一些关于词汇在序列中相对或绝对位置的信息。直观来说，如果语句中每一个词都有特定的位置，那么每一个词都可以使用向量编码位置信息。将这样的位置向量与词嵌入向量相结合，那么我们就为每一个词引入了一定的位置信息，注意力机制也就能分辨出不同位置的词。

谷歌研究者将「位置编码」添加到输入词嵌入中，位置编码有和词嵌入相同的维度 d_model，每一个词的位置编码与词嵌入向量相加可得出这个词的最终编码。目前有很多种位置编码，包括通过学习和固定表达式构建的。

在这一项实验中，谷歌研究者使用不同频率的正弦和预先函数：

![img](https://image.jiqizhixin.com/uploads/editor/37bb6cdb-b296-42f5-8ba3-c0f7bbc01682/1526094613695.png)

其中 pos 为词的位置，i 为位置编码向量的第 i 个元素。给定词的位置 pos，我们可以将词映射到 d_model 维的位置向量，该向量第 i 个元素就由上面两个式子计算得出。也就是说，位置编码的每一个维度对应于正弦曲线，波长构成了从 2π到 10000⋅2π的等比数列。

上面构建了绝对位置的位置向量，但词的相对位置同样非常重要，这也就是谷歌研究者采用三角函数表征位置的精妙之处。正弦与余弦函数允许模型学习相对位置，这主要根据两个变换：sin(α+β)=sinα cosβ+cosα sinβ 以及 cos(α+β)=cosα cosβ−sinα sinβ。

对于词汇间固定的偏移量 k，位置向量 PE(pos+k) 可以通过 PE(pos) 与 PE(k) 的组合表示，这也就表示了语言间的相对位置。

以下定义了位置编码，其中我们对词嵌入与位置编码向量的和使用 Dropout，默认可令_drop=0.1。div_term 实现的是分母，而 pe[:, 0::2] 表示第二个维度从 0 开始以间隔为 2 取值，即偶数。

```
class PositionalEncoding(nn.Module):
 "Implement the PE function."
 def __init__(self, d_model, dropout, max_len=5000):
 super(PositionalEncoding, self).__init__()
 self.dropout = nn.Dropout(p=dropout)

 # Compute the positional encodings once in log space.
 pe = torch.zeros(max_len, d_model)
 position = torch.arange(0, max_len).unsqueeze(1)
 div_term = torch.exp(torch.arange(0, d_model, 2) *
 -(math.log(10000.0) / d_model))
 pe[:, 0::2] = torch.sin(position * div_term)
 pe[:, 1::2] = torch.cos(position * div_term)
 pe = pe.unsqueeze(0)
 self.register_buffer('pe', pe)

 def forward(self, x):
 x = x + Variable(self.pe[:, :x.size(1)], 
 requires_grad=False)
 return self.dropout(x)
```

以下将基于一个位置将不同的正弦曲线添加到位置编码向量中，曲线的频率和偏移量在每个维度上都不同。

```
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
```

![img](https://image.jiqizhixin.com/uploads/editor/c6daf9df-06db-40ae-9951-ed14afead370/1526094614221.png)

谷歌等研究者在原论文中表示他们同样对基于学习的位置编码进行了实验，并发现这两种方法会产生几乎相等的结果。所以按照性价比，他们还是选择了正弦曲线，因为它允许模型在训练中推断更长的序列。

**模型整体**

下面，我们定义了一个函数以构建模型的整个过程，其中 make_model 在输入原语词汇表和目标语词汇表后会构建两个词嵌入矩阵，而其它参数则会构建整个模型的架构。

```
def make_model(src_vocab, tgt_vocab, N=6, 
 d_model=512, d_ff=2048, h=8, dropout=0.1):
 "Helper: Construct a model from hyperparameters."
 c = copy.deepcopy
 attn = MultiHeadedAttention(h, d_model)
 ff = PositionwiseFeedForward(d_model, d_ff, dropout)
 position = PositionalEncoding(d_model, dropout)
 model = EncoderDecoder(
 Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
 Decoder(DecoderLayer(d_model, c(attn), c(attn), 
 c(ff), dropout), N),
 nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
 nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
 Generator(d_model, tgt_vocab))

 # This was important from their code. 
 # Initialize parameters with Glorot / fan_avg.
 for p in model.parameters():
 if p.dim() > 1:
 nn.init.xavier_uniform(p)
 return model
```

在以上的代码中，make_model 函数将调用上面我们定义的各个模块，并将它们组合在一起。我们会将 Multi-Head Attention 子层、全连接子层和位置编码等结构传入编码器与解码器主体函数，再根据词嵌入向量与位置编码向量完成输入与标注输出的构建。以下简单地示例了如何使用 make_model 函数构建模型：

```
# Small example model.
tmp_model = make_model(10, 10, 2)
None
```

**训练**

这一部分将描述模型的训练方案。首先需要介绍一些训练标准编码器解码器模型的工具，例如定义一个批量的目标以储存原语序列与目标语序列，并进行训练。前文的模型架构与函数定义我们主要参考的原论文，而后面的具体训练过程则主要参考了 Alexander 的实现经验。

**批量和掩码**

以下定义了保留一个批量数据的类，并且它会使用 Mask 在训练过程中限制目标语的访问序列。

```
class Batch:
 "Object for holding a batch of data with mask during training."
 def __init__(self, src, trg=None, pad=0):
 self.src = src
 self.src_mask = (src != pad).unsqueeze(-2)
 if trg is not None:
 self.trg = trg[:, :-1]
 self.trg_y = trg[:, 1:]
 self.trg_mask = \
 self.make_std_mask(self.trg, pad)
 self.ntokens = (self.trg_y != pad).data.sum()

 @staticmethod
 def make_std_mask(tgt, pad):
 "Create a mask to hide padding and future words."
 tgt_mask = (tgt != pad).unsqueeze(-2)
 tgt_mask = tgt_mask & Variable(
 subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
 return tgt_mask
```

我们下一步需要创建一般的训练和评分函数，以持续追踪损失的变化。在构建一般的损失函数后，我们就能根据它更新参数。

如下定义了训练中的迭代循环，我们使用 loss_compute() 函数计算损失函数，并且每运行 50 次迭代就输出一次训练损失，这有利于监控训练情况。

```
def run_epoch(data_iter, model, loss_compute):
 "Standard Training and Logging Function"
 start = time.time()
 total_tokens = 0
 total_loss = 0
 tokens = 0
 for i, batch in enumerate(data_iter):
 out = model.forward(batch.src, batch.trg, 
 batch.src_mask, batch.trg_mask)
 loss = loss_compute(out, batch.trg_y, batch.ntokens)
 total_loss += loss
 total_tokens += batch.ntokens
 tokens += batch.ntokens
 if i % 50 == 1:
 elapsed = time.time() - start
 print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
 (i, loss / batch.ntokens, tokens / elapsed))
 start = time.time()
 tokens = 0
 return total_loss / total_tokens
```

**训练数据与分批**

Alexander 等人的模型在标准的 WMT 2014 英语-德语数据集上进行训练，这个数据集包含 450 万条语句对。语句已经使用双字节编码（byte-pair encoding）处理，且拥有约为 37000 个符号的原语-目标语共享词汇库。对于英语-法语的翻译任务，. 原论文作者使用了更大的 WMT 2014 英语-法语数据集，它包含 3600 万条语句，且将符号分割为包含 32000 个 word-piece 的词汇库。

原论文表示所有语句对将一同执行分批操作，并逼近序列长度。每一个训练批量包含一组语句对，大约分别有 25000 个原语词汇和目标语词汇。

Alexander 等人使用 torch text 进行分批，具体的细节将在后面讨论。下面的函数使用 torchtext 函数创建批量数据，并确保批量大小会填充到最大且不会超过阈值（使用 8 块 GPU，阈值为 25000）。

```
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
 "Keep augmenting batch and calculate total number of tokens + padding."
 global max_src_in_batch, max_tgt_in_batch
 if count == 1:
 max_src_in_batch = 0
 max_tgt_in_batch = 0
 max_src_in_batch = max(max_src_in_batch, len(new.src))
 max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
 src_elements = count * max_src_in_batch
 tgt_elements = count * max_tgt_in_batch
 return max(src_elements, tgt_elements)
```

batch_size_fn 将抽取批量数据，且每一个批量都抽取最大原语序列长度和最大目标语序列长度，如果长度不够就使用零填充增加。

**硬件与策略**

原论文在一台机器上使用 8 块 NVIDIA P100 GPU 训练模型，基本模型使用了论文中描述的超参数，每一次迭代大概需要 0.4 秒。基本模型最后迭代了 100000 次，共花了 12 个小时。而对于大模型，每一次迭代需要花 1 秒钟，所以训练 300000 个迭代大概需要三天半。但我们后面的真实案例并不需要使用如此大的计算力，因为我们的数据集相对要小一些。

**优化器**

原论文使用了 Adam 优化器，其中β_1=0.9、 β_2=0.98 和 ϵ=10^{−9}。在训练中，研究者会改变学习率为 l_rate=d−0.5model⋅min(step_num−0.5,step_num⋅warmup_steps−1.5)。

学习率的这种变化对应于在预热训练中线性地增加学习率，然后再与迭代数的平方根成比例地减小。这种 1cycle 学习策略在实践中有非常好的效果，一般使用这种策略的模型要比传统的方法收敛更快。在这个实验中，模型采用的预热迭代数为 4000。注意，这一部分非常重要，我们需要以以下配置训练模型。

```
class NoamOpt:
 "Optim wrapper that implements rate."
 def __init__(self, model_size, factor, warmup, optimizer):
 self.optimizer = optimizer
 self._step = 0
 self.warmup = warmup
 self.factor = factor
 self.model_size = model_size
 self._rate = 0

 def step(self):
 "Update parameters and rate"
 self._step += 1
 rate = self.rate()
 for p in self.optimizer.param_groups:
 p['lr'] = rate
 self._rate = rate
 self.optimizer.step()

 def rate(self, step = None):
 "Implement `lrate` above"
 if step is None:
 step = self._step
 return self.factor * \
 (self.model_size ** (-0.5) *
 min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
 return NoamOpt(model.src_embed[0].d_model, 2, 4000,
 torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

使用不同模型大小和最优化超参数下的变化曲线：

![img](https://image.jiqizhixin.com/uploads/editor/6ecac8ae-e1bb-4752-8660-d32089b5668e/1526094613784.png)

```
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
 NoamOpt(512, 1, 8000, None),
 NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
```

**正则化**

- 标签平滑

在训练中，Alexander 等人使用了标签平滑的方法，且平滑值ϵ_ls=0.1。这可能会有损困惑度，因为模型将变得更加不确定它所做的预测，不过这样还是提升了准确度和 BLEU 分数。

Harvard NLP 最终使用 KL 散度实现了标签平滑，与其使用 one-hot 目标分布，他们选择了创建一个对正确词有置信度的分布，而其它平滑的概率质量分布将贯穿整个词汇库。

```
class LabelSmoothing(nn.Module):
 "Implement label smoothing."
 def __init__(self, size, padding_idx, smoothing=0.0):
 super(LabelSmoothing, self).__init__()
 self.criterion = nn.KLDivLoss(size_average=False)
 self.padding_idx = padding_idx
 self.confidence = 1.0 - smoothing
 self.smoothing = smoothing
 self.size = size
 self.true_dist = None

 def forward(self, x, target):
 assert x.size(1) == self.size
 true_dist = x.data.clone()
 true_dist.fill_(self.smoothing / (self.size - 2))
 true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
 true_dist[:, self.padding_idx] = 0
 mask = torch.nonzero(target.data == self.padding_idx)
 if mask.dim() > 0:
 true_dist.index_fill_(0, mask.squeeze(), 0.0)
 self.true_dist = true_dist
 return self.criterion(x, Variable(true_dist, requires_grad=False))
```

下面，我们可以了解到概率质量如何基于置信度分配到词。

```
# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
 [0, 0.2, 0.7, 0.1, 0], 
 [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
 Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None
```

![img](https://image.jiqizhixin.com/uploads/editor/e9610f87-1600-4957-8bdb-7407f24fcc16/1526094614641.png)

标签平滑实际上在模型对某些选项非常有信心的时候会惩罚它。

```
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
 d = x + 3 * 1
 predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
 ])
 #print(predict)
 return crit(Variable(predict.log()),
 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
```

![img](https://image.jiqizhixin.com/uploads/editor/039a4867-b7d9-419b-b245-8bfd78d289d8/1526094614097.png)

**简单的序列翻译案例**

我们可以从简单的复制任务开始尝试。若从小词汇库给定输入符号的一个随机集合，我们的目标是反向生成这些相同的符号。

```
def data_gen(V, batch, nbatches):
 "Generate random data for a src-tgt copy task."
 for i in range(nbatches):
 data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
 data[:, 0] = 1
 src = Variable(data, requires_grad=False)
 tgt = Variable(data, requires_grad=False)
 yield Batch(src, tgt, 0)
```

**计算模型损失**

```
class SimpleLossCompute:
 "A simple loss compute and train function."
 def __init__(self, generator, criterion, opt=None):
 self.generator = generator
 self.criterion = criterion
 self.opt = opt

 def __call__(self, x, y, norm):
 x = self.generator(x)
 loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
 y.contiguous().view(-1)) / norm
 loss.backward()
 if self.opt is not None:
 self.opt.step()
 self.opt.optimizer.zero_grad()
 return loss.data[0] * norm
```

**贪婪解码**

```
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
 torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
 model.train()
 run_epoch(data_gen(V, 30, 20), model, 
 SimpleLossCompute(model.generator, criterion, model_opt))
 model.eval()
 print(run_epoch(data_gen(V, 30, 5), model, 
 SimpleLossCompute(model.generator, criterion, None)))
Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
1.9274832487106324
Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
1.657595729827881
Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
1.4888023376464843
Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
0.9853351473808288
Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
0.34273059368133546
```

这些代码将简单地使用贪婪解码预测译文。

```
def greedy_decode(model, src, src_mask, max_len, start_symbol):
 memory = model.encode(src, src_mask)
 ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
 for i in range(max_len-1):
 out = model.decode(memory, src_mask, 
 Variable(ys), 
 Variable(subsequent_mask(ys.size(1))
 .type_as(src.data)))
 prob = model.generator(out[:, -1])
 _, next_word = torch.max(prob, dim = 1)
 next_word = next_word.data[0]
 ys = torch.cat([ys, 
 torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
 return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
 1 2 3 4 5 6 7 8 9 10
[torch.LongTensor of size 1x10]
```

**真实案例**

现在，我们将使用 IWSLT 德语-英语数据集实现翻译任务。该任务要比论文中讨论的 WMT 任务稍微小一点，但足够展示整个系统。我们同样还展示了如何使用多 GPU 处理来令加速训练过程。

```
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
```

**数据加载**

我们将使用 torchtext 和 spacy 加载数据集，并实现分词。

```
# For data loading.
from torchtext import data, datasets

if True:
 import spacy
 spacy_de = spacy.load('de')
 spacy_en = spacy.load('en')

 def tokenize_de(text):
 return [tok.text for tok in spacy_de.tokenizer(text)]

 def tokenize_en(text):
 return [tok.text for tok in spacy_en.tokenizer(text)]

 BOS_WORD = '<s>'
 EOS_WORD = '</s>'
 BLANK_WORD = "<blank>"
 SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
 TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
 eos_token = EOS_WORD, pad_token=BLANK_WORD)

 MAX_LEN = 100
 train, val, test = datasets.IWSLT.splits(
 exts=('.de', '.en'), fields=(SRC, TGT), 
 filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
 len(vars(x)['trg']) <= MAX_LEN)
 MIN_FREQ = 2
 SRC.build_vocab(train.src, min_freq=MIN_FREQ)
 TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```

我们希望有非常均匀的批量，且有最小的填充，因此我们必须对默认的 torchtext 分批函数进行修改。这段代码修改了默认的分批过程，以确保我们能搜索足够的语句以找到紧凑的批量。

**数据迭代器**

迭代器定义了分批过程的多项操作，包括数据清洗、整理和分批等。

```
class MyIterator(data.Iterator):
 def create_batches(self):
 if self.train:
 def pool(d, random_shuffler):
 for p in data.batch(d, self.batch_size * 100):
 p_batch = data.batch(
 sorted(p, key=self.sort_key),
 self.batch_size, self.batch_size_fn)
 for b in random_shuffler(list(p_batch)):
 yield b
 self.batches = pool(self.data(), self.random_shuffler)

 else:
 self.batches = []
 for b in data.batch(self.data(), self.batch_size,
 self.batch_size_fn):
 self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
 "Fix order in torchtext to match ours"
 src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
 return Batch(src, trg, pad_idx)
```

**多 GPU 训练**

最后为了快速训练，我们使用了多块 GPU。这段代码将实现多 GPU 的词生成，但它并不是针对 Transformer 的具体方法，所以这里并不会具体讨论。多 GPU 训练的基本思想即在训练过程中将词生成分割为语块（chunks），并传入不同的 GPU 实现并行处理，我们可以使用 PyTorch 并行基元实现这一点。

```
replicate - split modules onto different gpus.
scatter - split batches onto different gpus
parallel_apply - apply module to batches on different gpus
gather - pull scattered data back onto one gpu.
nn.DataParallel - a special module wrapper that calls these all before evaluating.
# Skip if not interested in multigpu.
class MultiGPULossCompute:
 "A multi-gpu loss compute and train function."
 def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
 # Send out to different gpus.
 self.generator = generator
 self.criterion = nn.parallel.replicate(criterion, 
 devices=devices)
 self.opt = opt
 self.devices = devices
 self.chunk_size = chunk_size

 def __call__(self, out, targets, normalize):
 total = 0.0
 generator = nn.parallel.replicate(self.generator, 
 devices=self.devices)
 out_scatter = nn.parallel.scatter(out, 
 target_gpus=self.devices)
 out_grad = [[] for _ in out_scatter]
 targets = nn.parallel.scatter(targets, 
 target_gpus=self.devices)

 # Divide generating into chunks.
 chunk_size = self.chunk_size
 for i in range(0, out_scatter[0].size(1), chunk_size):
 # Predict distributions
 out_column = [[Variable(o[:, i:i+chunk_size].data, 
 requires_grad=self.opt is not None)] 
 for o in out_scatter]
 gen = nn.parallel.parallel_apply(generator, out_column)

 # Compute loss. 
 y = [(g.contiguous().view(-1, g.size(-1)), 
 t[:, i:i+chunk_size].contiguous().view(-1)) 
 for g, t in zip(gen, targets)]
 loss = nn.parallel.parallel_apply(self.criterion, y)

 # Sum and normalize loss
 l = nn.parallel.gather(loss, 
 target_device=self.devices[0])
 l = l.sum()[0] / normalize
 total += l.data[0]

 # Backprop loss to output of transformer
 if self.opt is not None:
 l.backward()
 for j, l in enumerate(loss):
 out_grad[j].append(out_column[j][0].grad.data.clone())

 # Backprop all loss through transformer. 
 if self.opt is not None:
 out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
 o1 = out
 o2 = nn.parallel.gather(out_grad, 
 target_device=self.devices[0])
 o1.backward(gradient=o2)
 self.opt.step()
 self.opt.optimizer.zero_grad()
 return total * normalize
```

下面，我们利用前面定义的函数创建了模型、度量标准、优化器、数据迭代器和并行化：

```
# GPUs to use
devices = [0, 1, 2, 3]
if True:
 pad_idx = TGT.vocab.stoi["<blank>"]
 model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
 model.cuda()
 criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
 criterion.cuda()
 BATCH_SIZE = 12000
 train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
 repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
 batch_size_fn=batch_size_fn, train=True)
 valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
 repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
 batch_size_fn=batch_size_fn, train=False)
 model_par = nn.DataParallel(model, device_ids=devices)
None
```

下面可以训练模型了，Harvard NLP 团队首先运行了一些预热迭代，但是其它的设定都能使用默认的参数。在带有 4 块 Tesla V100 的 AWS p3.8xlarge 中，批量大小为 12000 的情况下每秒能运行 27000 个词。

**训练系统**

```
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
if False:
 model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
 torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
 for epoch in range(10):
 model_par.train()
 run_epoch((rebatch(pad_idx, b) for b in train_iter), 
 model_par, 
 MultiGPULossCompute(model.generator, criterion, 
 devices=devices, opt=model_opt))
 model_par.eval()
 loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
 model_par, 
 MultiGPULossCompute(model.generator, criterion, 
 devices=devices, opt=None))
 print(loss)
else:
 model = torch.load("iwslt.pt")
```

一旦训练完成了，我们就能解码模型并生成一组翻译，下面我们简单地翻译了验证集中的第一句话。该数据集非常小，所以模型通过贪婪搜索也能获得不错的翻译效果。

```
for i, batch in enumerate(valid_iter):
 src = batch.src.transpose(0, 1)[:1]
 src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
 out = greedy_decode(model, src, src_mask, 
 max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
 print("Translation:", end="\t")
 for i in range(1, out.size(1)):
 sym = TGT.vocab.itos[out[0, i]]
 if sym == "</s>": break
 print(sym, end =" ")
 print()
 print("Target:", end="\t")
 for i in range(1, batch.trg.size(0)):
 sym = TGT.vocab.itos[batch.trg.data[i, 0]]
 if sym == "</s>": break
 print(sym, end =" ")
 print()
 break
Translation: <unk> <unk> . In my language , that means , thank you very much . 
Gold: <unk> <unk> . It means in my language , thank you very much . 
```

**实验结果**

在 WMT 2014 英语到法语的翻译任务中，原论文中的大型的 Transformer 模型实现了 41.0 的 BLEU 分值，它要比以前所有的单模型效果更好，且只有前面顶级的模型 1/4 的训练成本。在 Harvard NLP 团队的实现中，OpenNMT-py 版本的模型在 EN-DE WMT 数据集上实现了 26.9 的 BLEU 分值。