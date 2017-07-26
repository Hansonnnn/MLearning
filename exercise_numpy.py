# -*- coding:utf-8 -*-
import numpy as np
from numpy import pi

a  = np.arange(15).reshape(3,5)
print a
print type(a),a.shape
print a.ndim, a.size
"""
此时的a的类型已经变成ndarray的类型。
ndarray.ndim 返回ndarray的数组维度，如上的数组a在创建时以二维形式创建，则返回值即为2
ndarray.shape 返回数组的实际情况，如以上a数组的维度为3行5列的数组
ndarray.size
ndarray.dtype 
ndarray.itemsize
ndarray.data
"""
"""创建array"""
b = np.array([1,2,4])
# b = np.array(1,2,4)
"""使用numpy正确创建数组的方式如上，而array一般只会接受一个或duoge
元素列表为参数，而不是将每个单独的元素作为参数传入"""
print b
print type(b)

c = np.array([[1,2],[3,4]],dtype=complex)
print c

"""
arange的另外一种用法，在指定第三个参数的时候，arange提供了为我们
生成由前两个参数限定的范围内的列表，并且列表内的元素相邻元素之差为指
定的第三个参数值，如以下例子
"""
print np.arange(0,2,0.3)

print np.linspace(0,2,9)

x = np.linspace(0,2*pi,100)
f = np.sin(x)
print f

d = np.array([20,30,40,50])
e = np.arange(4)

print d * e

A = np.array([[1,0],[2,4]])

B = np.array([[2,1],[3,1]])

print A * B

print A.dot(B)

print np.dot(A,B)

c = np.array([ 1.        ,  2.57079633,  4.14159265])
print c
print c.dtype.name

d = np.exp(c*1j)

print d.dtype.name
print d

h = np.arange(12).reshape(3,4)

print h
print h.sum(axis=0)
print h.max(axis=0)
print h.min(axis=1)







