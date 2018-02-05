#!/usr/local/bin/lua


require "torch"
require "cutorch"

function add(a,b)
	return a, a*b
end

a = torch.ones(5,2)
print("a = ", a)
b = torch.Tensor(2,5):fill(4)
print("b = ", b)
print(add(a,b))
a = a:cuda()
b = b:cuda() --转为cuda运算的tensor

