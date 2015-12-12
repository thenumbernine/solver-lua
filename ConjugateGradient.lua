--[[
source: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
args:
	A = linear function A : x -> x
	b = solution vector
	x0 (optional) = initial guess
	clone = vector clone
	dot = vector dot
	MInv = inverse of preconditioner linear function MInv : x -> x
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000
	
	local bNorm = dot(b,b)
	if bNorm == 0 then bNorm = 1 end

	local x = clone(args.x0 or b)
	local r = b - A(x)
	local MInvR = MInv(r)
	local rDotMInvR = dot(r, MInvR)
	
	local err = dot(r, r) / bNorm
	if errorCallback and errorCallback(err, 0) then return x end
	if err < epsilon then return x end
	
	local p = clone(MInvR)
	for iter=1,maxiter do
		local Ap = A(p)
		local alpha = rDotMInvR / dot(p, Ap)
		x = x + p * alpha
		r = r - Ap * alpha
		
		local err = dot(r, r) / bNorm
		if errorCallback and errorCallback(err, iter) then break end
		if err < epsilon then break end
		
		MInvR = MInv(r)
		local nRDotMInvR = dot(r, MInvR)
		local beta = nRDotMInvR / rDotMInvR
		p = MInvR + p * beta
		
		rDotMInvR = nRDotMInvR
	end
	return x
end
