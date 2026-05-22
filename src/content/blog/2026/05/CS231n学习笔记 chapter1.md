---
title: "CS231n学习笔记 chapter1"
categories: deeplearning
tags:  
  - DeepLearning
  - CS231n
  - 学习笔记
id: "CS231n_c1"
date: 2026-05-22 23:21:07
cover: "/public/assets/images/cs231n.png"
recommend: true
---

:::note
CS231n是斯坦福大学推出的包括CV，深度学习，生成模型等内容的优质课程，这里是我2026年学习该课程的学习笔记，供大家参考！
:::

## chapter1  线性分类器进行图像处理

### 困难和挑战

+ **视角变化**：同一个物体，摄像机可以从多个角度观察
+ **大小变化**：真实的物体的大小是会变化的
+ **形变**：很多东西的形状会发生很大的变化
+ **遮挡**：目标物体可能被挡住，有时候只有物体的一部分是可见的
+ **光照条件**：在像素层面上，光照的影响非常大
+ **背景干扰**：物体可能混入背景之中
+ **类内差异**：一类物体的个体之间差异过大

### 数据驱动方法

给计算机很多数据，然后实现学习算法，让计算机学习到每个类的外形，从而实现图像分类

### 图像分类的流程

图像分类就是输入一个元素为像素值的数组，然后给他分配一个分类标签：

+ 输入：包含N个图像的集合，每个图像的标签是K种分类标签中的一种，成为训练集
+ 学习：使用训练集让计算机学习每个类到底长什么样（训练分类器或学习一个模型）
+ 评价：让分类器来预测它没有见过的图像的分类标签，并以此来评价分类器的质量。

### Nearest Neighbor分类器

![img](https://pic3.zhimg.com/fff49fd8cec00f77f657a4c4a679b030_r.jpg)

由图，左边是来自数据库的样本图像，右边第一列是测试图像，后面是根据NN算法，从训练集中选出的10张最类似的图片

比较的方法就是逐个像素比较，最后将差异值全部加起来。也就是将两张图片转换为向量I~1~,I~2~，然后计算他们的L1距离：
$$
d_1(I_1, I_2)=\sum_{p}\left|I_1^p - I_2^p\right|
$$
以图片中的一个颜色通道为例来进行说明。两张图片使用L1距离来进行比较。逐个像素求差值，然后将所有差值加起来得到一个数值。如果两张图片一模一样，那么L1距离为0，但是如果两张图片很是不同，那L1值将会非常大。

![img](https://pic4.zhimg.com/95cfe7d9efb83806299c218e0710a6c5_r.jpg)

下面，用代码来实现这个分类器。首先，将数据加载到内存中，并分成4个数组：训练数据和标签，测试数据和标签。在下面的代码中，**Xtr**（大小是50000x32x32x3）存有训练集中所有的图像，**Ytr**是对应的长度为50000的1维数组，存有图像对应的分类标签（从0到9）：

```python
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```

然后实现一个分类器：

```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```

使用L1距离实现分类器：

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

### 距离选择

计算向量间的距离有很多种方法，另一个常用的方法是**L2距离**，从几何学的角度，可以理解为它在计算两个向量间的欧式距离。L2距离的公式如下：
$$
d_2(I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}
$$

### k-Nearest Neighbor分类器

与其只找最相近的那1个图片的标签，我们找最相似的k个图片的标签，然后让他们针对测试图片进行投票，最后把票数最高的标签作为对测试图片的预测。所以当k=1的时候，k-Nearest Neighbor分类器就是Nearest Neighbor分类器。从直观感受上就可以看到，更高的k值可以让分类的效果更平滑，使得分类器对于异常值更有抵抗力。

![img](https://pica.zhimg.com/51aef845faa10195e33bdd4657592f86_r.jpg)

### 超参数

k-NN分类器需要设定k值，那么选择哪个k值最合适的呢？我们可以选择不同的距离函数，比如[L1范数]和[L2范数]等，那么选哪个好？还有不少选择我们甚至连考虑都没有考虑到（比如：点积）。所有这些选择，被称为**超参数（hyperparameter）**。

> 不能使用测试集来确定超参数，因为会导致过拟合

从训练集中取出一部分数据用来调优，我们称之为**验证集（validation set）**

**交叉验证**。有时候，训练集数量较小（因此验证集的数量更小），人们会使用一种被称为**交叉验证**的方法，这种方法更加复杂些。用一个例子，如果是交叉验证集，我们就不是取1000个图像，而是将训练集平均分成5份，其中4份用来训练，1份用来验证。然后我们循环着取其中4份来训练，其中1份来验证，最后取所有5次验证结果的平均值作为算法验证结果。

![img](https://pic3.zhimg.com/6a3ceec60cc0a379b4939c37ee3e89e8_1440w.png)

此图表明了在课上的例子中，k=7表现最优

### NN分类器的优劣

Nearest Neighbor分类器易于理解，实现简单。其次，算法的训练不需要花时间，因为其训练过程只是将训练集数据存储起来。然而测试要花费大量时间计算，因为每个测试图像需要和所有存储的训练图像进行比较，这显然是一个缺点。在实际应用中，我们关注测试效率远远高于训练效率。其实，我们后续要学习的卷积神经网络在这个权衡上走到了另一个极端：虽然训练花费很多时间，但是一旦训练完成，对新的测试数据进行分类非常快。这样的模式就符合实际使用需求。



狗的图片可能和青蛙的图片非常接近，这是因为两张图片都是白色背景。从理想效果上来说，我们肯定是希望同类的图片能够聚集在一起，而不被背景或其他不相关因素干扰。