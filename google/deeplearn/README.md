### 逻辑回归分类器

一种线性分类器，接受输入后并对输入执行一个线性函数来生成预测。线性函数实际上是巨大的矩阵相乘，如下

```
 WX + b =Y

```
线性函数将输入当作一个矢量（X表示），然后与一个矩阵相乘产生预测，每个类一个输出。在线性函数当中，我们将
W看作为权重，b看作为偏置项，而机器学习的重点就在于W与b，我们需要通过不断的训练尝试来寻找合适的权重与偏置。

