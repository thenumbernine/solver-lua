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
	local b = assert(args.b)
	local errorCallback = args.errorCallback
	local clone = args.clone
	local dot = args.dot
	local scale = args.scale
	local invScale = args.invScale
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000

	b = clone(b)
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
