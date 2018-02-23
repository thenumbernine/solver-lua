#!/usr/bin/env luajit
require 'ext'
local cl = require 'cl'
local matrix = require 'matrix'
local gnuplot = require 'gnuplot'

local n = 128

local env = require 'cl.obj.env'{size={n,n}, precision='float'}

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

local A = env:kernel{
	argsOut = {{name='y', type='global real*', buffer=true}},
	argsIn = {{name='x', type='const global real*', buffer=true}},
	body = [[
	if (i.x == 0 || 
		i.y == 0 ||
		i.x >= size.x ||
		i.y >= size.y)
	{
		y[index] = x[index];
		return;
	}
	
	const real hSq = .01;
	y[index] = -(
		x[index + stepsize.x]
		+ x[index - stepsize.x]
		+ x[index + stepsize.y]
		+ x[index - stepsize.y]
		- 4. * x[index]
	) / hSq;
]],
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
	--'conjres',
	'bicgstab',
	--'gmres',
} do
	require('solver.cl.'..solver){
		env = env,
		A = A,
		b = b,
		x = x,
		errorCallback = print,
		restart = 10,
	}()
	splot(x, 'x-'..solver)
end

