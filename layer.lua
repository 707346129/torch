#!/usr/local/bin/lua


require "torch";
require "nn";


m = nn.SpatialConvolution(1,3,2,2) -- learn 3 2x2 kernels
print(m.weight) -- initially, the weights are randomly initialized

print(m.bias) 
-- The operation in a convolution layer is: output = convolution(input,weight) + bias