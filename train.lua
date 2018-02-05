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

