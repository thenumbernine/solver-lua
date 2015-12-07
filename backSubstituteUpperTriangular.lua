-- solves for x in the system y = u x for upper triangular matrix u
return function(u, y)
	local n = #y
	assert(#u == n)
	assert(#u[1] == n)
	local x = {}
	for i=n,1,-1 do
		local sum = 0
		for j=i+1,n do
			sum = sum + x[j] * u[i][j]
		end
		x[i] = (y[i] - sum) / u[i][i]
	end
	return x
end
