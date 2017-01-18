#!/usr/bin/env luajit
local env = require 'cl.obj.env'{size=4}
local ffi = require 'ffi'

local x = env:buffer{type='real4', data=ffi.new('real4[1]', {{s={-1,-1,-1,0}}})}

local f = env:kernel{
	argsOut = {{name='y', type='real4', obj=true}},
	argsIn = {{name='x', type='real4', obj=true}},
	body = [[ 
	//where did that cross product go?
	y->s0 = 0;	
	y->s1 = -x->s2;
	y->s2 = x->s0 - 1.;
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
		errorCallback = function(err, iter)
			print('gmres err',err,'iter',iter)
		end,
	},
}
jfnk()

local ptr = x:toCPU()
print('x', ptr[0].s0, ptr[0].s1, ptr[0].s2, ptr[0].s3)
