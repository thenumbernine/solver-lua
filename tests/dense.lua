#!/usr/bin/env luajit
local range = require 'ext.range'
local table = require 'ext.table'
local matrix = require 'matrix'
local backsub = require 'solver.backsub'
local fwdsub = require 'solver.fwdsub'


-- forward substitute test
do
	local errs = table()
	for n=1,10 do
		for trial=1,10 do
			local a = matrix.lambda({n,n}, function(i,j) return i<j and 0 or math.random(10) - 5 end)
			local y = matrix.lambda({n}, function(i) return math.random(10) - 5 end)
			local x = matrix(fwdsub(a, y))
			local y_ = a * x
			local err = (y - y_):norm()
			errs:insert(err)
		end
	end
	print('fwdsub','err',(errs:sup()))
end

-- back substitute test
do
	local errs = table()
	for n=1,10 do
		for trial=1,10 do
			local a = matrix.lambda({n,n}, function(i,j) return i>j and 0 or math.random(10) - 5 end)
			local y = matrix.lambda({n}, function(i) return math.random(10) - 5 end)
			local x = matrix(backsub(a, y))
			local y_ = a * x
			local err = (y - y_):norm()
			errs:insert(err)
		end
	end
	print('backsub','err',(errs:sup()))
end

local function rebuildQR(q, r)
	return matrix(q) * matrix(r)
end
local function rebuildLU(l,u,p)
	return matrix(l) * matrix(u)
end
local function rebuildLUP(l,u,p)
	return matrix(p):transpose() * matrix(l) * matrix(u)
end

--[[
solve a x = b for x
let a = q r
q r x = b
r x = q^t b
--]]
local function solveLinearQR(b, q, r)
	local qtb = matrix(q):transpose() * matrix(b)
	return backsub(r, qtb)
end

--[[
pt l u x = b
l u x = p b
ux = l^-1 p b
x = u^-1 l^-1 p b
--]]
local function solveLinearLUP(b, l, u, p)
	local pb = matrix(p) * matrix(b)
	local ux = fwdsub(l, pb)
	return backsub(u, ux)
end

local errs = table()
for n=1,5 do
	for m=n,n do	-- LU requires square, the others require m>=n for mxn matrices
		local a = matrix.lambda({m,n},function(i,j) return math.random() end)
		for _,solverInfo in ipairs{
			{name='qr_gramschmidt', solver=require 'solver.qr_gramschmidt', rebuild=rebuildQR, solveLinear=solveLinearQR},
			{name='qr_gramschmidt_classic', solver=require 'solver.qr_gramschmidt_classic', rebuild=rebuildQR, solveLinear=solveLinearQR},
			{name='qr_householder', solver=require 'solver.qr_householder', rebuild=rebuildQR, solveLinear=solveLinearQR},
			--{name='lu', solver=require 'solver.lu', rebuild=rebuildLU, solveLinear=solveLinearLU},
			{name='lup', solver=require 'solver.lup', rebuild=rebuildLUP, solveLinear=solveLinearLUP},
		} do
			for _,test in ipairs{
				{
					name = 'rebuild',
					solve = function(a, solverInfo)
						local a_ = solverInfo.rebuild(solverInfo.solver(a))
						local err = (a_ - a):norm()
						return err
					end,
				},
				{
					name = 'inverse basis error',
					solve = function(a, solverInfo)
						if m ~= n then return end	-- only do square matrices
						print(solverInfo.name)
						print('a:')
						table.map(a, function(ai) print('['..table.concat(ai, ', ')..']') end)
						local err = 0
						for j=1,n do
							local a_j = range(n):map(function(i) return a[i][j] end)
							local aInv_j = solverInfo.solveLinear(a_j, solverInfo.solver(a))
							print('a^-1['..j..'] = ['..table.concat(aInv_j, ', ')..']')
							for i=1,n do
								err = err + math.abs(aInv_j[i] - (i==j and 1 or 0))
							end
						end
						print('err',err)
						return err
					end,
				},
			} do
				local err = test.solve(a, solverInfo)
				--print('|qr-a| ='..err)
				if not errs[solverInfo.name] then errs[solverInfo.name] = {} end
				if not errs[solverInfo.name][test.name] then errs[solverInfo.name][test.name] = 0 end
				errs[solverInfo.name][test.name] = math.max(errs[solverInfo.name][test.name], err)
			end
		end
	end
end

for solverName,solverErrInfo in pairs(errs) do
	for testName,errInfo in pairs(solverErrInfo) do
		print('solver', solverName, 'test', testName, 'err', errInfo)
	end
end
