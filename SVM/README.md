## SVM 支持向量机

>在机器学习中，支持向量机（英语：Support Vector Machine，常简称为SVM，又名支持向量网络[1]）
是在分类与回归分析中分析数据的监督式学习模型与相关的学习算法。给定一组训练实例，每个训练实例被标记为属于两个类别中的一个或另一个，
SVM训练算法创建一个将新的实例分配给两个类别之一的模型，使其成为非概率二元线性分类器。SVM模型是将实例表示为空间中的点，
这样映射就使得单独类别的实例被尽可能宽的明显的间隔分开。然后，将新的实例映射到同一空间，并基于它们落在间隔的哪一侧来预测所属类别。

### 多变量数据集

多变量数据集定义为与某一现象的不同相关联的一组属性值集合。

### 降维

模型维度是指数据集中独立的具体数量。为了降低模型的维度，需要通过降低维度并保证模型不失准确性和精度，这需要在降低维度时候
慎重选择最优的特征维度。执行降维时候需要使用以下几个过程。

* 特征选择：选择一个特征子集来获得更好的训练次数或者提升模型精度。

* 特征抽取：PCA和多维度度量(MDS)是两个典型的算法

* 降维：为防止使用多维度数据工作时出现数据对分析结果的影响（"维度的诅咒"），使用PCA或者LDA等方式解决


