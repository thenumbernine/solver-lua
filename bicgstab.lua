--[[
source: https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Preconditioned_BiCGSTAB
args:
	A = linear function A : x -> x
	AT = transpose of A
	b = solution vector
	x (optional) = initial guess, default to b
	rHat (optional) = alternative r, should be r dot rHat ~= 0, defaults to r
	zero = zero vector
	clone = vector clone function
	dot = vector dot function
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local AT = assert(args.AT)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone
	local MInvT = args.MInv and assert(args.MInvT, "you provided a MInv but not a MInvT") or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = 10000
	local zero = assert(args.zero)

	local b = clone(assert(args.b))
	local x = clone(args.x or b)
	local xStar = clone(x)
	local r = b - A(x)

	local rHat = clone(args.rHat or r)	-- pick some rHat such that r dot rHat ~= 0 

	local err = dot(r,r)
	if errorCallback and errorCallback(err, 0) then return x end
	if err < epsilon then return x end

	local rho = 1
	local alpha = 1
	local omega = 1

	local p = clone(zero)
	local v = clone(zero)

	for iter=1,maxiter do
		local nrho = dot(rHat, r)
		local beta = nrho / rho * alpha / omega
		local np = r + beta * (p - omega * v)
		local y = KInv(np)
		local nv = A(y)
		alpha = nrho / dot(rHat, nv)
		local s = r - alpha * nv
		local s2 = dot(s,s)
		if s2 < epsilon then
			x = x + alpha * p
			break
		end
		local z = KInv(s)
		local t = A(z)
		local K1InvT = K1Inv(t)
		local nomega = dot(K1InvT, K1Inv(s)) / dot(K1InvT, K1InvT)
		local nx = x + alpha * y + nomega * z
		-- TODO "if x is accurate enough then quit"
		local nr = s - nomega * t
		-- TODO errorCallback

		rho = nrho
		p = np
		v = nv
		omega = nomega
		x = nx
		r = nr
	end
end
