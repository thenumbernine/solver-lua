#!/usr/bin/env luajit
local range = require 'ext.range'
local table = require 'ext.table'
local matrix = require 'matrix'

-- because i'm still working on my matrix class ...
local function matmul(a,b)
	local m = #a
	local n = #a[1]
	local p = #b
	local q = #b[1]
	assert(n == p, "matrix sizes don't match")
	local c = {}
	for i=1,m do
		c[i] = {}
		for j=1,q do
			local sum = 0
			for k=1,n do
				sum = sum + a[i][k] * b[k][j]
			end
			c[i][j] = sum
		end
	end
	return c
end

local n = 5

local a = matrix.lambda(5,5,function(i,j) return math.random() end)
print('a=\n'..a)

for _,info in ipairs{
	{name='GramSchmidt', solver=require 'LinearSolvers.GramSchmidt'},
	{name='GramSchmidtClassical', solver=require 'LinearSolvers.GramSchmidtClassical'},
} do
	print(info.name)
	local q, r = info.solver(a)
	q, r = matrix(q), matrix(r)
	print('q=\n'..q)
	print('r=\n'..r)

	print('qr=\n'..matrix(matmul(q,r)))
	print('qr-a=\n'..(matmul(q,r)-a))
	print('|qr-a| ='..(matmul(q,r)-a):norm())
end
