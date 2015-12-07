#!/usr/bin/env luajit
local range = require 'ext.range'
local table = require 'ext.table'
local matrix = require 'matrix'
local backSubstituteUpperTriangular = require 'LinearSolvers.backSubstituteUpperTriangular'
local forwardSubstituteLowerTriangular = require 'LinearSolvers.forwardSubstituteLowerTriangular'

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
solve a x = y for x
let a = q r 
q r x = y
r x = q^t y
--]]
local function solveLinearQR(y, q, r)
	local qty = matrix(q):transpose() * matrix(y)
	return backSubstituteUpperTriangular(r, qty)
end

--[[
l (u p x) = y
u (p x) = (l^-1 y)
x = p^t u^-1 l^-1 y
--]]
local function solveLinearLUP(y, l, u, p)
	local lInvY = forwardSubstituteLowerTriangular(l, y)
	local px = backSubstituteUpperTriangular(u, lInvY)
	return matrix(p):transpose() * matrix(px)
end

local errs = table()
for n=1,5 do
	for m=n,n do	-- LU requires square, the others require m>=n for mxn matrices
		local a = matrix.lambda({m,n},function(i,j) return math.random() end)
		for _,solverInfo in ipairs{
			{name='GramSchmidt', solver=require 'LinearSolvers.GramSchmidt', rebuild=rebuildQR, solveLinear=solveLinearQR},
			{name='GramSchmidtClassical', solver=require 'LinearSolvers.GramSchmidtClassical', rebuild=rebuildQR, solveLinear=solveLinearQR},
			{name='HouseholderQR', solver=require 'LinearSolvers.HouseholderQR', rebuild=rebuildQR, solveLinear=solveLinearQR},
			--{name='LUDecomposition', solver=require 'LinearSolvers.LUDecomposition', rebuild=rebuildLU, solveLinear=solveLinearLU},
			{name='LUPDecomposition', solver=require 'LinearSolvers.LUPDecomposition', rebuild=rebuildLUP, solveLinear=solveLinearLUP},
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
