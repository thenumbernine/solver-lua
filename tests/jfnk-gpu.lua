#!/usr/bin/env luajit
local env = require 'cl.obj.env'{size=3}
local ffi = require 'ffi'

local x = env:buffer{type='real', data={-1,-1,-1}}

local f = env:kernel{
	argsOut = {{name='y', type='real', obj=true}},
	argsIn = {{name='x', type='real', obj=true}},
	body = [[ 
	//where did that cross product go?
	if (get_local_id(0) == 0) {
		y[0] = 0;	
		y[1] = -x[2];
		y[2] = x[1] - 1.;
	}
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
		errorCallback = function(err, iter, x)
			print('gmres err',err,'iter',iter)
		end,
	},
}
jfnk()

local ptr = x:toCPU()
print('x', require 'ext.range'(3):map(function(i) return ptr[i-1] end):concat', ')
