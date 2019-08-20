#!/usr/bin/env luajit
require 'ext'
local cl = require 'ffi.OpenCL'
local matrix = require 'matrix'
local gnuplot = require 'gnuplot'

local n = 128

local env = require 'cl.obj.env'{
	size = {n, n},
	precision = 'float',
	queue = {
		properties = cl.CL_QUEUE_PROFILING_ENABLE,
	},
}

-- A x = b ... solve for x
local b = env:buffer{name='b'}

env:kernel{
	argsOut = {b},
	body = [[
	b[index] = (
		i.x >= size.x/4 && i.x < size.x*3/4 &&
		i.y >= size.y/4 && i.y < size.y*3/4
	) ? 1. : 0.;
]],
}()

local event = require 'cl.event'()
local A = env:kernel{
	argsOut = {{name='y', type='global real*', buffer=true}},
	argsIn = {{name='x', type='const global real*', buffer=true}},
	body = [[
	//OOB
	if (i.x >= size.x || i.y >= size.y) return;	

	//boundary
	if (i.x == 0 || 
		i.y == 0 ||
		i.x == size.x-1 ||
		i.y == size.y-1)
	{
		y[index] = x[index];
		return;
	}

	//PDE
	const real hSq = .01;
	y[index] = -(
		x[index + stepsize.x]
		+ x[index - stepsize.x]
		+ x[index + stepsize.y]
		+ x[index - stepsize.y]
		- 4. * x[index]
	) / hSq;
]],
	event = event,
}

local function splot(gpubuf, name)
	local cpubuf = gpubuf:toCPU()
	gnuplot{
		output = 'krylov-cl-'..name..'.png',
		style = 'data lines',
		griddata = {
			x = matrix{n}:lambda(function(i) return i end),
			y = matrix{n}:lambda(function(i) return i end),
			matrix{n,n}:lambda(function(i,j) return cpubuf[(i-1) + n * (j-1)] end),
		},
		{splot=true, using='1:2:3', title=name},
	}
end

splot(b, 'b')

-- TODO returning 'x' doesn't return an env:buffer object
local x = env:buffer{name='x'}

for _,solver in ipairs{
	--'conjgrad',
	'conjres',
	--'bicgstab',
	--'gmres',
} do
	require('solver.cl.'..solver){
		env = env,
		A = A,
		b = b,
		x = x,
		errorCallback = function(res, iter, x)
			print(iter, res)
		end,
		restart = 10,
		maxiter = 1000,
	}()
	splot(x, 'x-'..solver)
end

env.cmds[1]:finish()
local start = event:getProfilingInfo'CL_PROFILING_COMMAND_START'
local fin = event:getProfilingInfo'CL_PROFILING_COMMAND_END'
print('duration', tonumber(fin - start)..' ns')
