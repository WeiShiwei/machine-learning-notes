​		**tf-idf**（英语：**t**erm **f**requency–**i**nverse **d**ocument **f**requency）是一种用于[信息检索](https://zh.wikipedia.org/wiki/資訊檢索)与[文本挖掘](https://zh.wikipedia.org/wiki/文本挖掘)的常用加权技术。tf-idf是一种统计方法，用以评估一字词对于一个文件集或一个[语料库](https://zh.wikipedia.org/wiki/語料庫)中的其中一份[文件](https://zh.wikipedia.org/wiki/文件)的重要程度。**字词的重要性随着它在文件中出现的次数成[正比](https://zh.wikipedia.org/wiki/正比)增加，但同时会随着它在语料库中出现的频率成反比下降。**tf-idf加权的各种形式常被[搜索引擎](https://zh.wikipedia.org/wiki/搜索引擎)应用，作为文件与用户查询之间相关程度的度量或评级。除了tf-idf以外，互联网上的搜索引擎还会使用基于链接分析的评级方法，以确定文件在搜索结果中出现的顺序。

## 目录



- [1原理](https://zh.wikipedia.org/wiki/Tf-idf#原理)
- [2例子](https://zh.wikipedia.org/wiki/Tf-idf#例子)
- [3在向量空间模型里的应用](https://zh.wikipedia.org/wiki/Tf-idf#在向量空間模型裡的應用)
- [4tf-idf的理论依据及不足](https://zh.wikipedia.org/wiki/Tf-idf#tf-idf的理论依据及不足)
- [5参考资料](https://zh.wikipedia.org/wiki/Tf-idf#參考資料)
- [6外部链接](https://zh.wikipedia.org/wiki/Tf-idf#外部連結)

## 原理[[编辑](https://zh.wikipedia.org/w/index.php?title=Tf-idf&action=edit&section=1)]

在一份给定的文件里，**词频**（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率。这个数字是对**词数**（term count）的归一化，以防止它偏向长的文件。（同一个词语在长文件里可能会比短文件有更高的词数，而不管该词语重要与否。）对于在某一特定文件里的词语{\displaystyle t_{i}}![t_{{i}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8b61e3d4d909be4a19c9a554a301684232f59e5a)来说，它的重要性可表示为：



以上式子中{\displaystyle n_{i,j}}![n_{{i,j}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/292ee1f4999efc9a7938840c62ea763f9489ddd7)是该词在文件{\displaystyle d_{j}}![d_{{j}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3fa3426b07cfa37c76382ddbecfb4c880889657f)中的出现次数，而分母则是在文件{\displaystyle d_{j}}![d_{{j}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/3fa3426b07cfa37c76382ddbecfb4c880889657f)中所有字词的出现次数之和。

**逆向文件频率**（inverse document frequency，idf）是一个词语普遍重要性的度量。某一特定词语的idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的[对数](https://zh.wikipedia.org/wiki/對數)得到：



其中

- |D|：语料库中的文件总数
- {\displaystyle |\{j:t_{i}\in d_{j}\}|}![|\{j:t_{{i}}\in d_{{j}}\}|](https://wikimedia.org/api/rest_v1/media/math/render/svg/84bc20901276fa6a1c4c679205c5771d608553b2)：包含词语{\displaystyle t_{i}}![t_{{i}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/8b61e3d4d909be4a19c9a554a301684232f59e5a)的文件数目（即{\displaystyle n_{i,j}\neq 0}![n_{{i,j}}\neq 0](https://wikimedia.org/api/rest_v1/media/math/render/svg/466698d7cd0e5e8a1ed60f6570a45dd8686f5c64)的文件数目）如果词语不在数据中，就导致分母为零，因此一般情况下使用{\displaystyle 1+|\{j:t_{i}\in d_{j}\}|}![1+|\{j:t_{{i}}\in d_{{j}}\}|](https://wikimedia.org/api/rest_v1/media/math/render/svg/7686335bc9835f6f3c5fb6403e9857bc7f0aa7e7)

然后



某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的tf-idf。因此，tf-idf倾向于过滤掉常见的词语，保留重要的词语。



tf-idf维基百科

https://zh.wikipedia.org/wiki/Tf-idf

