#!/usr/bin/env luajit
local env = require 'cl.obj.env'{size=1}
local ffi = require 'ffi'
local x = env:buffer{type='real4', data=ffi.new('real4[1]', {{s={-1,-1,-1,0}}})}

local f = env:kernel{
	argsOut = {name='y', type='real4', obj=true},
	argsIn = {name='x', type='real4', obj=true},
	body = [[ *y = cross( (real4)(1,0,0,0), *x) - (real4)(0,0,1,0);	]],
}

local jfnk = require 'solver.cl.jfnk'{
	env = env,
	x = x,
	f = f,
	errorCallback = function(err, iter)
		print('err',err,'iter',iter)
	end,
}
jfnk()
