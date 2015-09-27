#!/usr/bin/env luajit
local class = require 'ext.class'
local table = require 'ext.table'
local file = require 'ext.file'
local range = require 'ext.range'
local Jacobi = require 'LinearSolvers.Jacobi'
local ConjugateGradient = require 'LinearSolvers.ConjugateGradient'
local ConjugateResidual = require 'LinearSolvers.ConjugateResidual'

local vec = class()

function vec:init(n)
	if type(n) == 'number' then
		for i=1,n do
			self[i] = 0
		end
	elseif type(n) == 'table' then
		for i=1,#n do
			self[i] = n[i]
		end
	end
end

function vec.__add(a,b)
	local na = #a
	local nb = #b
	assert(na == nb)
	local n = na
	local c = vec(a)
	for i=1,n do
		c[i] = c[i] + b[i]
	end
	return c
end

function vec.__mul(a,b)
	local sa = type(a) == 'number'
	local sb = type(b) == 'number'
	assert(sa or sb)
	-- make sure b is the scalar
	if sa then
		a,b = b,a
		sa,sb = sb,sa
	end
	return vec(table.map(a, function(ai) return b * ai end))
end

function vec.__unm(a)
	return -1 * a
end

function vec.__sub(a,b)
	return a + -b
end

function vec.dot(a,b)
	local na = #a
	local nb = #b
	assert(na == nb)
	local n = na
	local sum = 0
	for i=1,n do
		sum = sum + a[i] * b[i]
	end
	return sum
end

function vec:norm()
	return vec.dot(self,self)
end

function vec:__tostring()
	return '{'..table.concat(self, ', ')..'}'
end

local function matfunc(A)
	-- y_i = A_ij x_j
	return function(x)
		local y = vec(#A)
		for i=1,#A do
			assert(#x == #A[i])
			y[i] = vec.dot(A[i], x)
		end
		return y
	end
end

os.remove('cr.txt')
os.remove('cg.txt')
for _,problem in ipairs{
	-- works with jacobi:
--	{A={{1,.07,0,0,0},{-.07,1,.07,0,0},{0,-.07,1,.07,0},{0,0,-.07,1,.07},{0,0,0,-.07,1}}, b={1,2,3,4,5}},
	-- doesn't:
--	{A={{1,2},{3,4}}, b={5,6}},
--	{A={{1,2,3},{4,5,6},{7,8,9}}, b={100,200,300}},
--	{A={{1,1},{2,1}}, b={2,0}}
	

	(function()
		local n = 16
		local A = {}
		local b = {}
		for i=1,n do
			b[i] = i<n/2 and 0 or 1
		end
		for i=1,n do
			A[i] = {}
			for j=1,n do
				A[i][j] = 0
			end
			A[i][i] = 2/4
			if i>1 then A[i][i-1] = 1/4 end
			if i<n then A[i][i+1] = 1/4 end
		end
		return {A=A, b=b}
	end)(),
} do
	local A,b = problem.A,problem.b
	local fA = matfunc(A)
	local fMInv = function(x)
		x = vec(x)
		for i=1,#x do
			x[i] = x[i] / A[i][i]
		end
		return x
	end

--[[
	fA = function(x)
		local y = vec(#x)	--make a vector as big as x is
		for i=2,m-1 do
			for j=2,n-1 do
				k=j+n*i
				y[k] = x[k] + x[k-1] + x[k+1] + x[k+n] + x[k-n]
			end
		end
		return y
	end
--]]

--[[
preconditioners:
MInv(x) = M^-1 * x for M the preconditioning matrix: a matrix such that M^-1 * A has a smaller condition number than A alone
	condition number: sigmaMax(A) / sigmaMin(A) for sigmaMax & sigmaMin the max & min singular values of A, for singular values the eigenvalues of A^* A
popular preconditioner options:
	Jacobi preconditioner: M = diag(a_ii)
	SPAI: M minimizes ||A M^-1 - I||_F for ||.||_F the Frobenius norm
--]]

	local errors = table()
	local x = ConjugateGradient{
		A = fA,
		MInv = fMInv,
		b = b,
		errorCallback = function(err) errors:insert(err) end,
		clone = vec,
		norm = vec.norm,
		dot = vec.dot,
	}
	print(x)
	file['cg.txt'] = errors:concat'\n'

	local errors = table()
	local x = ConjugateResidual{
		A = fA,
		b = b,
		errorCallback = function(err) errors:insert(err) end,
		clone = vec,
		norm = vec.norm,
		dot = vec.dot,
	}
	print(x)
	file['cr.txt'] = errors:concat'\n'

	local ADiag = range(#A):map(function(i) return A[i][i] end)
	ADiag = vec(ADiag)
	
	local errors = table()
	local x = Jacobi{
		A = fA,
		b = b,
		ADiag = ADiag,
		errorCallback = function(err) errors:insert(err) end,
		clone = vec,
		norm = vec.norm,
		dot = vec.dot,
		scale = function(a,b)
			local c = vec(a)
			for i=1,#c do
				c[i] = c[i] * b[i]
			end
			return c
		end,
		invScale = function(a,b)
			local c = vec(a)
			for i=1,#c do
				c[i] = c[i] / b[i]
			end
			return c
		end,
	}
	print(x)
	file['jacobi.txt'] = errors:concat'\n'
end

