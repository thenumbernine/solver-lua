--[[
source: https://en.wikipedia.org/wiki/Conjugate_residual_method#Preconditioning
args:
	A = linear function A : x -> x
	b = solution vector
	x0 (optional) = initial guess
	clone = vector clone
	dot = vector dot
	norm = vector norm (defaults to dot(x,x))
	MInv = inverse of preconditioner linear function MInv : x -> x
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)

this method applies MInv to r, then computes norms, etc
the conjugate gradient applies MInv as a weighted norm of r (i.e. only applies once between r's rather than twice) 
which is better?
which is correct?

--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local norm = args.norm or function(a) return dot(a,a) end
	local MInv = args.MInv or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000

	b = clone(b)
	local x = clone(args.x0 or b)
	local r = MInv(b - A(x))
	
	local r2 = norm(r)
	if errorCallback and errorCallback(r2, 0) then return x end
	if r2 < epsilon then return x end
	
	local Ar = A(r)
	local rAr = dot(r, Ar)
	local p = clone(r)
	local Ap = A(p)
	for iter=1,maxiter do
		local alpha = rAr / dot(Ap, MInv(Ap))	-- dot() could be replaced by weightedNorm() ...
		x = x + p * alpha
		local nr = r - MInv(Ap) * alpha
		local nAr = A(nr)
		local nrAr = dot(nr, nAr)
		local beta = nrAr / rAr
		
		local nr2 = norm(nr)
		if errorCallback and errorCallback(nr2, iter) then break end
		if nr2 < epsilon then break end
		
		r = nr
		rAr = nrAr
		Ar = nAr
		p = r + p * beta
		Ap = Ar + Ap * beta
	end
	return x
end
