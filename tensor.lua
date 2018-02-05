#!/usr/local/bin/lua


require "torch"
b = {}
a = torch.Tensor(5,3)
print("a = ")
print(a)
print("b = ")
print(b)
a = torch.rand(5,3)
print("a = ")
print(a)
b = torch.rand(3,4)
c = a * b
d = torch.Tensor(5,4)
d:mm(a,b)
print("c = ") 
print(c)
print("d = ") 
print(d)
p = torch.Tensor{1,2,3,4,5}
print(p)
p:reshape(5,1) --返回一个变形后的tensor，p本身不变
print(p)
p = p:reshape(5,1) --p变形，需要变形为矩阵才能参与运算
print(p)
r = torch.range(3,10)
print(r)
