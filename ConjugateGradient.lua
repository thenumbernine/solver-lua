--[[
args:
	A = linear function A : x -> x
	b = solution vector
	x0 (optional) = initial guess
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
	local x = clone(args.x0 or b)
	local r = b - A(x)
	local r2 = norm(r)
	if errorCallback and errorCallback(r2, 0) then return x end
	if r2 < epsilon then return x end
	local p = clone(r)
	for iter=1,maxiter do
		local Ap = A(p)
		local alpha = r2 / dot(p, Ap)
		x = x + p * alpha
		local nr = r - Ap * alpha
		local nr2 = norm(nr)
		local beta = nr2 / r2
		if errorCallback and errorCallback(nr2, iter) then break end
		if nr2 < epsilon then break end
		r = nr
		r2 = nr2
		p = r + p * beta
	end
	return x
end


