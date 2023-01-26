local math = require 'ext.math'

--[[
source: https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
args:
	A = linear function A : x -> x
	b = solution vector
	x (optional) = initial guess
	clone = vector clone
	dot = vector dot
	MInv = inverse of preconditioner linear function MInv : x -> x
	errorCallback (optional) = function(|r|, iteration, x)
		returns true if iterations should be stopped
	epsilon (optional)
	maxiter (optional)

b and x and A() use __sub, __add, __mul
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

	--local bSq = dot(b,b)	-- TODO normalize err by bSq?
	local x = clone(args.x or b)
	local r = b - A(x)
	local MInvR = MInv(r)
	local rDotMInvR = dot(r, MInvR)

	local rSq = dot(r, r)
	local err = math.sqrt(rSq)
	if errorCallback and errorCallback(err, 0, x) then return x end
	if not math.isfinite(err) or err < epsilon then return x end

	local p = clone(MInvR)
	for iter=1,maxiter do
		local Ap = A(p)
		local alpha = rDotMInvR / dot(p, Ap)
		x = x + p * alpha
		r = r - Ap * alpha

		rSq = dot(r, r)
		err = math.sqrt(rSq)
		if errorCallback and errorCallback(err, iter, x) then break end
		if not math.isfinite(err) or err < epsilon then break end

		MInvR = MInv(r)
		local nRDotMInvR = dot(r, MInvR)
		local beta = nRDotMInvR / rDotMInvR
		p = MInvR + p * beta

		rDotMInvR = nRDotMInvR
	end
	return x
end
