--[[
args:
	A = linear function A : x -> x
	b = vector
	clone = vector clone
	dot = vector dot
	norm = vector norm (defaults to dot(x,x))
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local norm = args.norm or function(a) return dot(a,a) end
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000
	
	b = clone(b)
	local x = clone(b)
	local r = b - A(x)
	local r2 = norm(r)
	if errorCallback and errorCallback(r2) then return x end
	if r2 < epsilon then return x end
	local Ar = A(r)
	local rAr = dot(r, Ar)
	local p = clone(r)
	local Ap = A(p)
	for iter=1,maxiter do
		local alpha = rAr / norm(Ap)
		x = x + p * alpha
		local nr = r - Ap * alpha
		local Anr = A(nr)
		local nrAr = dot(nr, Anr)
		local beta = nrAr / rAr
		local nr2 = norm(nr)
		if errorCallback and errorCallback(nr2) then break end
		if nr2 < epsilon then break end
		r = nr
		rAr = nrAr
		Ar = Anr
		p = r + p * beta
		Ap = Ar + Ap * beta
	end
	return x
end


