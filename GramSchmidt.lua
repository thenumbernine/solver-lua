--[[
Algorithm 8.1 from Thefethen and Bau "Numerical Liner Algebra"

input:
	a = m x n table of numbers
output:
	q, r decomposition
--]]
return function(a)
	local m = #a
	local n = #a[1]
	local v = {}
	local r = {}
	local q = {}
	for j=1,m do
		v[i] = {}
		r[i] = {}
		q[i] = {}
		for i=1,n do
			v[i][j] = a[i][j]
			r[i][j] = 0
			q[i][j] = 0
		end
	end
	for i=1,n do
		local norm = 0
		for j=1,m do
			norm = norm + v[j][i] * v[j][i]
		end
		r[i][i] = math.sqrt(norm)
		for j=1,m do
			q[j][i] = v[j][i] / r[i][i]
		end
		for j=i+1,n do
			local sum = 0
			for k=1,m do
				sum = sum + q[k][i] * v[k][j]
			end
			r[i][j] = sum
			for k=1,m do
			v[k][j] = v[k][j] - r[i][j] * q[k][i]
		end
	end
	return q, r
end
