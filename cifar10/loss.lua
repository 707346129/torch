#!/usr/local/bin/lua


require "torch";
require "nn";

criterion = nn.ClassNLLCriterion()