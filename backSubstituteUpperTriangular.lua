-- solves for x in the system b = a x for upper triangular matrix a
return function(a, b)
	local m = #a
	local n = #a[1]
	assert(m == #b)
	assert(m >= n)
	local x = {}	-- size n
	for i=n,1,-1 do
		local sum = 0
		for j=i+1,n do
			sum = sum + x[j] * a[i][j]
		end
		x[i] = (b[i] - sum) / a[i][i]
	end
	return x
end
