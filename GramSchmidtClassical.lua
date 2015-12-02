--[[
Algorithm 7.1 from Thefethen and Bau "Numerical Liner Algebra"

input:
	a = n x n table of numbers
output:
	q, r decomposition
--]]
return function(a)
	local m = #a
	local n = #a[1]
	local r = {}
	local v = {}
	local q = {}
	for i=1,m do
		r[i] = {}
		v[i] = {}
		q[i] = {}
		for j=1,n do
			r[i][j] = 0
			v[i][j] = 0
			q[i][j] = 0
		end
	end
	for j=1,n do
		for k=1,m do
			v[k][j] = a[k][j]
		end
		for i=1,j-1 do
			local sum = 0
			for k=1,m do
				sum = sum + q[k][i] * a[k][j]
			end
			r[i][j] = sum 
			for k=1,m do
				v[k][j] = v[k][j] - r[i][j] * q[k][i]
			end
		end
		local sum = 0
		for k=1,m do
			sum = sum + v[k][j] * v[k][j]
		end
		r[j][j] = math.sqrt(sum)
		for k=1,m do
			q[k][j] = v[k][j] / r[j][j]
		end
	end
	return q, r
end
