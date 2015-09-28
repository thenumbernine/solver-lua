--[[
source: https://en.wikipedia.org/wiki/Biconjugate_gradient_method#The_algorithm
args:
	A = linear function A : x -> x
	AT = transpose of A
	b = solution vector
	x0 (optional) = initial guess
	clone = vector clone
	dot = vector dot
	MInv = inverse of preconditioner linear function MInv : x -> x
	MInvT = transpose application of inverse of preconditioner linear function
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
notice: wiki says this method is numerically unstable.  so why did I even bother implement it?
--]]
return function(args)
	local A = assert(args.A)
	local AT = assert(args.AT)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone
	local MInvT = args.MInv and assert(args.MInvT, "you provided a MInv but not a MInvT") or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = 10000

	b = clone(b)
	local x = clone(args.x0 or b)
	local xStar = clone(x)
	local r = b - A(x)
	local MInvR = MInv(r)
	local rStar = bStar - AT(xStar)
	local rStarMInvR = dot(rStar, MInvR)
	
	local err = dot(r,r)
	if errorCallback and errorCallback(err, 0) then return x end
	if err < epsilon then return x end

	local p = clone(MInvR)
	local pStar = MInvT(rStar)
	
	for iter=1,maxiter do
		local Ap = A(p)
		local ATPStar = AT(pStar)
		
		local alpha = rStarMInvR / dot(pStar, Ap)	-- dot(pStar, Ap) == dot(ATPStar, p) ... either one
		local nx = x + p * alpha
		local nxStar = xStar + pStar * alpha
		local nr = r - Ap * alpha
		local nrStar = rStar - ATPStar * alpha
	
		local MInvNR = MInv(nr)
		local MInvTNRStar = MInvT(nrStar)
		local nrStarMInvNR = dot(nrStar, MInvNR)	-- or dot(MInvTNRStar, nr)
		
		local err = dot(nr,nr)
		if errorCallback and errorCallback(err, iter) then break end
		if err < epsilon then break end
		
		local beta = nrStarMInvNR / rStarMInvR
		local np = MInvNR + p * beta
		local npStar = MInvTNRStar + pStar * beta
		
		x = nx
		xStar = nxStar
		r = nr
		rStar = nrStar
		rStarMInvR = nrStarMInvNR
		p = np
		pStar = npStar
	end
end
