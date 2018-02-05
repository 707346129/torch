#!/usr/local/bin/lua


require "torch"


function add(a,b)
	return a, a*b
end

a = torch.ones(5,2)
print("a = ", a)
b = torch.Tensor(2,5):fill(4)
print("b = ", b)
print(add(a,b))

