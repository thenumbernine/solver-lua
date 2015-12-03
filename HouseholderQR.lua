--[[
Algorithm 10.1 from Trefethen and Bau "Numerical Linear Algebra"

input:
	a = m x n table of numbers, m >= n
output:
	q, r decomposition
--]]
local unpack = unpack or table.unpack
local function applyQ(a,k,jmin,jmax,v)
	local m = #a
	for j=jmin,jmax do
		local vDotMj = 0
		for i=1,m-k+1 do
			vDotMj = vDotMj + v[i] * a[i+k-1][j]
		end
		for i=k,m do
			a[i][j] = a[i][j] - (2 * vDotMj) * v[i-k+1]
		end
	end

end
return function(a)
	local m = #a
	local n = #a[1]
	do
		local _a = {}
		for i=1,m do
			_a[i] = {unpack(a[i],1,n)}
		end
		a = _a
	end
	local qt = {}
	for i=1,m do
		qt[i] = {}
		for j=1,m do
			qt[i][j] = i == j and 1 or 0
		end
	end
	for k=1,n do
		local v = {}
		local vSq = 0
		for i=k,m do
			local a_ik = a[i][k]
			v[i-k+1] = a_ik
			vSq = vSq + a_ik * a_ik
		end
		v[1] = v[1] + math.sqrt(vSq) * (v[1] < 0 and -1 or 1)
		local vSq = 0
		for i=1,#v do
			local v_i = v[i]
			vSq = vSq + v_i * v_i
		end
		local vLen = math.sqrt(vSq)
		for i=1,#v do
			v[i] = v[i] / vLen
		end
		applyQ(a,k,k,n,v)
		applyQ(qt,k,1,m,v)
	end
	local q = {}
	for i=1,m do
		q[i] = {}
		for j=1,m do
			q[i][j] = qt[j][i]
		end
	end
	return q, a
end
