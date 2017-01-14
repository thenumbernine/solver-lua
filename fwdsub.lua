--[[
l x = y
solve for x
l is lower triangular
--]]
return function(l, y)
	local n = #y
	assert(#l == n)
	assert(#l[1] == n)
	local x = {}
	for i=1,n do
		local sum = 0
		for j=1,i-1 do
			sum = sum + l[i][j] * x[j]
		end
		x[i] = (y[i] - sum) / l[i][i]
	end
	return x
end
