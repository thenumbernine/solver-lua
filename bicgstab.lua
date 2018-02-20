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
	M1Inv (optional) preconditioner (inverse of 'K1' in the article)
	MInv (optional) preconditioner (inverse of 'K' in the article ... K = K1 K2)
--]]
return function(args)
	local A = assert(args.A)
	local AT = assert(args.AT)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone
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
		local y = MInv and MInv(np) or np
		local nv = A(y)
		alpha = nrho / dot(rHat, nv)
		local s = r - alpha * nv
		local s2 = dot(s,s)
		if s2 < epsilon then
			x = x + alpha * p
			break
		end
		local z = MInv and MInv(s) or s
		local t = A(z)
		local M1InvS = M1Inv and M1Inv(s) or s
		local M1InvT = M1Inv and M1Inv(t) or t
		local nomega = dot(M1InvT, M1InvS) / dot(M1InvT, M1InvT)
		local nx = x + alpha * y + nomega * z
		-- TODO "if x is accurate enough then quit"
		local nr = s - nomega * t
		-- TODO errorCallback
	
		local err = dot(nr, nr)
		if errorCallback and errorCallback(err, iter, nx) then return nx end
		if err < epsilon then	-- update approximation
			return nx
		end

		rho = nrho
		p = np
		v = nv
		omega = nomega
		x = nx
		r = nr
	end

	return x
end
