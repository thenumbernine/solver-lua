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
	assert(n == p, "matrix sizes don't match.  tried to multiply "..m..' x '..n..' with '..p..' x '..q)
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

local function transpose(a)
	local m = #a
	local n = #a[1]
	local t = {}
	for i=1,n do
		t[i] = {}
		for j=1,m do
			t[i][j] = a[j][i]
		end
	end
	return t
end
local function rebuildQR(q, r)
	return matmul(q,r)
end
local function rebuildLUP(l,u,p)
	return matmul(transpose(p),matmul(l,u))
end

local errs = table()
for n=1,100,5 do
	for m=n,n do	-- LU requires square, the others require m>=n for mxn matrices
		local a = matrix.lambda(m,n,function(i,j) return math.random() end)
		--print('a=\n'..a)

		for _,info in ipairs{
			{name='GramSchmidt', solver=require 'LinearSolvers.GramSchmidt', rebuild=rebuildQR},
			{name='GramSchmidtClassical', solver=require 'LinearSolvers.GramSchmidtClassical', rebuild=rebuildQR},
			{name='HouseholderQR', solver=require 'LinearSolvers.HouseholderQR', rebuild=rebuildQR},
			{name='LUDecomposition', solver=require 'LinearSolvers.LUDecomposition', rebuild=rebuildLUP},
		} do
			local a_ = info.rebuild(info.solver(a))
			local err = (a_ - a):norm()
			--print('|qr-a| ='..err)
			if not errs[info.name] then errs[info.name] = {} end
			errs[info.name].min = math.min(errs[info.name].min or err, err)
			errs[info.name].max = math.max(errs[info.name].max or err, err)
		end
	end
end

for name,info in pairs(errs) do
	print(name, 'min', info.min, 'max', info.max)
end
