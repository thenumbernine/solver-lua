--[[
Algorithm 10.1 from Trefethen and Bau "Numerical Linear Algebra"

input:
	a = m x m table of numbers
output:
	l,u,p decomposition
--]]
local unpack = unpack or table.unpack
return function(a)
	local m = #a
	local u = {}
	local l = {}
	local p = {}
	for i=1,m do
		u[i] = {unpack(a[i],1,m)}
		l[i] = {}
		for j=1,m do
			l[i][j] = i == j and 1 or 0
		end
		p[i] = {unpack(l[i],1,m)}
	end
	for k=1,m-1 do
		local i=k
		local abs_u_i_k = u[k][k]
		for j=k+1,m do
			local abs_u_j_k = math.abs(u[j][k])
			if abs_u_j_k > abs_u_i_k then
				i = j
				abs_u_i_k = abs_u_j_k
			end
		end
		for j=k,m do
			u[k][j], u[i][j] = u[i][j], u[k][j]
		end
		for j=1,k-1 do
			l[k][j], l[i][j] = l[i][j], l[k][j]
		end
		for j=1,m do
			p[k][j], p[i][j] = p[i][j], p[k][j]
		end
		for j=k+1,m do
			l[j][k] = u[j][k] / u[k][k]
			for i=k,m do
				u[j][i] = u[j][i] - l[j][k] * u[k][i]
			end
		end
	end
	return l,u,p
end
