#!/usr/bin/env luajit
local env = require 'cl.obj.env'()

local _3 = env:domain{size=3}
local x = _3:buffer{type='real', data={-1,-1,-1}}

local f = _3:kernel{
	argsOut = {{name='y', type='real', obj=true}},
	argsIn = {{name='x', type='real', obj=true}},
	body = [[
	//where did that cross product go?
	if (index == 0) y[0] = 0;
	if (index == 1) y[1] = -x[2];
	if (index == 2) y[2] = x[1] - 1.;
]],
}

local jfnk = require 'solver.cl.jfnk'{
	env = env,
	x = x,
	f = f,
	errorCallback = function(err, iter)
		print('jfnk err',err,'iter',iter)
	end,
	gmres = {
		errorCallback = function(err, iter, x_)
			print('gmres err',err,'iter',iter)
		end,
	},
}
jfnk()

local ptr = x:toCPU()
print('x', require 'ext.range'(3):map(function(i) return ptr[i-1] end):concat', ')
