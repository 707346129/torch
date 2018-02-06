#!/usr/local/bin/lua


require "torch";
require "nn";
require "network";
input = torch.rand(1,32,32)
output = net:forward(input)
print(output)
net:zeroGradParameters();
gradInput = net:backward(input, torch.rand(10))

print(#gradInput)

criterion = nn.ClassNLLCriterion() --损失函数
criterion:forward(output, 3) --3 为正确的 label
gradients = criterion:backward(output, 3)

gradInput = net:backward(input, gradients) --在最后一层后加入损失函数后重新计算grad

net:updateParameters(0.01)