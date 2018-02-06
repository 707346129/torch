#!/usr/local/bin/lua


require "torch";
require "nn";
require "loss"
require "model"
require "data"

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)