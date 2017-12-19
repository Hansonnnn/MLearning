### numpy学习

#### 基本常用

```
import numpy as np

a  = np.arange(15).reshape(3,5)
print a
print type(a),a.shape
print a.ndim, a.size

```
a是由numpy构造的ndarray类型的数组。

* ndarray.ndim 返回ndarray的数组维数，如上的数组a在创建时以二维形式创建，则返回值即为2

* ndarray.shape 数组的尺寸，如以上a数组的维度为3行5列的数组

* ndarray.size 数组的大小

* ndarray.dtype 数组元素类型，如a中元素的类型为int64

* ndarray.itemsize 数组的每个元素的大小（以字节为单位）

* ndarray.data  该缓冲区包含数组的实际元素。

#### 创建array

```
b = np.array([1,2,4]) #right

b = np.array(1,2,4)  #wrong

```
使用numpy的array方法创建数组的方式如上，而array一般只会接受一个或多个
列表为参数，但不接受将每个单独的元素作为参数传入，如上代码所示，其一是正确的创建方式，其二错误的将单个元素作为参数传入其中。

也可以在创建时指定其类型。

```
c = np.array([[1,2],[3,4]],dtype=complex)

```
arange的另外一种用法，在指定第三个参数的时候，arange提供了为我们
生成由前两个参数限定的范围内的列表，并且列表内的元素相邻元素之差为指
定的第三个参数值，如以下例子：

```
prinrt np.arange(5,30,5)
print np.arange(0,2,0.3)

```
当使用浮点参数时，由于有限的浮点精度，通常不可能预测获得的元素数量，因此可以使用linspace函数来获取我们所需的元素季哥元素个数，
如下所示

```
print np.linspace(0,2,9)

```
#### 基本的操作

* 加、减、乘、平方、比较大小

```
a = np.array([20,30,40,50])
b = np.arange(4)

c = a + b
c = a - b
c = a * b
c = b ** 2
a<35

```

* 矩阵相乘

numpy中`*`符号会对列表中每个元素逐个相乘计算得出结果，而对于数学中的矩阵相乘，numpy特意准备了矩阵乘法的功能函数dot。

```
A = np.array([[1,0],[2,4]])

B = np.array([[2,1],[3,1]])

print A * B

[[2 0]
 [6 4]]

```

```
A = np.array([[1,0],[2,4]])

B = np.array([[2,1],[3,1]])

print A.dot(B)

print np.dot(A,B)

[[ 2  1]
 [16  6]]

```

* 向上转型

在数组中出现多种不同类型的元素（比如小位元素与大位元素）或者数组计算后的结果更为复杂，比如精度更高。需要使用到向上转型数组类型

来存储其中的元素。

```
c = np.array([ 1.        ,  2.57079633,  4.14159265])
print c
print c.dtype.name

d = np.exp(c*1j)

print d.dtype.name
print d

```

* 功能函数

numpy中对于其数组带有一些可以直接调用的数学功能函数。如下示例

```
h = np.arange(1,5,2)

print h
print h.sum()
print h.max()
print h.min()

```
以上示例只是简单的一维维度的计算，根据源码观察，这些提供的功能函数是支持多维当中的计算。
如下对sum函数的源码研究

```
    def sum(self, axis=None, dtype=None, out=None): # real signature unknown; restored from __doc__
        """
        a.sum(axis=None, dtype=None, out=None)

            Return the sum of the array elements over the given axis.

            Refer to `numpy.sum` for full documentation.

            See Also
            --------
            numpy.sum : equivalent function
        """
        pass
```

```
h = np.arange(12).reshape(3,4)

print h
print h.sum(axis=0)
print h.max(axis=0)
print h.min(axis=1)

```

* 通用方法

numpy 在数据分析或数据挖掘当中也可以提供一些功能性函数，比如：

np.std() 求某一维度的方差

np.mean() 求某一维度的平均值

np.hstack()  叠加两个维度的数据为一个维度，比如

```
a = np.arange(0,12,3).reshape(-1,1)
print a

b = np.arange(0,16,4).reshape(-1,1)
print b

print np.hstack((a,b))


[[0]
 [3]
 [6]
 [9]]


 [[ 0]
 [ 4]
 [ 8]
 [12]]


[[ 0  0]
 [ 3  4]
 [ 6  8]
 [ 9 12]]
```

np.histogram() 计算某一维度的直方图用来观察数据分布。
















