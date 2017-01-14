local qr_householder = require 'solver.qr_householder'
local backsub = require 'solver.backsub'

--[[
solve a x = b for x
a = m * n table of numbers
b = solution vector of numbers
solver = function that accepts a row-major matrix 'a'
		and returns unitary 'q' and upper triangular 'r'

let a = q r 
q r x = b
r x = q^t b
--]]
return function(a, b, solver)
	solver = solver or qr_householder
	local q, r = solver(a)
	-- qtb = q^t * b 
	local m = #q[1]
	local n = #q
	local qtb = {}
	for i=1,m do
		local sum = 0
		for j=1,n do
			sum = sum + q[j][i] * b[j]
		end
		qtb[i] = sum
	end
	return backsub(r, qtb)
end
