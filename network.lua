#!/usr/local/bin/lua


require "torch";
require "nn";

net = nn.Sequential() --构建网络
--Concat为多路交织，Parallel为并行
net:add(nn.SpatialConvolution(1,6,5,5)) 
--添加卷基层(输入通道数，输出通道数，窗口长，宽，步长，步宽，pad长，宽)
--module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) 

net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
--(窗口长宽，步长长宽，pad长宽)
net:add(nn.SpatialConvolution(6,16,5,5)) 
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))  --转化为1维tensor
net:add(nn.Linear(16*5*5, 120)) --全连接输入400输出120
net:add(nn.ReLU())
net:add(nn.Linear(120,84))
net:add(nn.ReLU())
net:add(nn.Linear(84,10))
net:add(nn.LogSoftMax())
print('Lenet5\n'..net:__tostring())
