local LUPDecomposition = require 'LinearSolvers.LUPDecomposition'
local backSubstituteUpperTriangular = require 'LinearSolvers.backSubstituteUpperTriangular'
local forwardSubstituteLowerTriangular = require 'LinearSolvers.forwardSubstituteLowerTriangular'

--[[
solve a x = b for x
l u p = a
pt l u x = b
l u x = p b
ux = l^-1 p b
x = u^-1 l^-1 p b
--]]
return function(a, b, solver)
	local l, u, p = LUPDecomposition(a)
	-- TODO return a sparse representation of 'p' so we can permute in O(n) instead of O(n^2)
	local pb = {}
	for i=1,#p do
		local sum = 0
		for j=1,#p[1] do
			sum = sum + p[i][j] * b[j]
		end
		pb[i] = sum
	end
	local ux = forwardSubstituteLowerTriangular(l, pb)
	return backSubstituteUpperTriangular(u, ux)
end
