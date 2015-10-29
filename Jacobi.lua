--[[
source: https://en.wikipedia.org/wiki/Jacobi_method#Algorithm
args:
	A = linear function A : x -> x
	b = solution to Ax=b
	x (optional) = initial guess.  uses b by default.
	clone = vector clone
	dot = vector dot
	scale = vector*vector scale
	invScale = vector/vector inverse scale
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local errorCallback = args.errorCallback
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local scale = assert(args.scale)
	local invScale = assert(args.invScale)
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000

	local b = clone(assert(args.b))
	local ADiag = clone(assert(args.ADiag))
	local x = clone(args.x or b)
	for iter=1,maxiter do
		local nx = b - A(x)
		nx = nx + scale(ADiag, x)	-- remove diagonal
		nx = invScale(nx, ADiag)		-- divide by diagonal
		local r = b - A(nx)
		local r2 = dot(r, r)
		if errorCallback and errorCallback(r2, iter) then break end
		if r2 < epsilon then break end
		x = nx
	end
	return x
end
