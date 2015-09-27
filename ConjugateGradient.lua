--[[
args:
	A = linear function A : x -> x
	b = solution vector
	x0 (optional) = initial guess
	clone = vector clone
	dot = vector dot
	MInv = preconditioner linear function MInv : x -> x
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone	-- preconditioner
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000

	b = clone(b)
	local x = clone(args.x0 or b)
	local r = b - A(x)
	local z = MInv(r)
	local rDotZ = dot(r, z)
	if errorCallback and errorCallback(rDotZ, 0) then return x end
	if rDotZ < epsilon then return x end
	local p = clone(z)
	for iter=1,maxiter do
		local Ap = A(p)
		local alpha = rDotZ / dot(p, Ap)
		x = x + p * alpha
		local nr = r - Ap * alpha
		local nz = MInv(nr)
		local nRDotZ = dot(nr, nz)
		local beta = nRDotZ / rDotZ
		if errorCallback and errorCallback(nRDotZ, iter) then break end
		if nRDotZ < epsilon then break end
		r = nr
		z = nz
		rDotZ = nRDotZ
		p = r + p * beta
	end
	return x
end
