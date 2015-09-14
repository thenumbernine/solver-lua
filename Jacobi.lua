--[[
args:
	A = linear function A : x -> x
	b = vector
	clone = vector clone
	dot = vector dot
	norm = vector norm (defaults to dot(x,x))
	scale = vector*vector scale
	invScale = vector/vector inverse scale
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)
--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local ADiag = assert(args.ADiag)
	local errorCallback = args.errorCallback
	local clone = args.clone
	local dot = args.dot
	local norm = args.norm or function(a) return dot(a,a) end
	local scale = args.scale
	local invScale = args.invScale
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 10000

	b = clone(b)
	A_diag = clone(A_diag)
	local x = clone(b)
	for iter=1,maxiter do
		local nx = b - A(x)
		nx = nx + scale(ADiag, x)	-- remove diagonal
		nx = invScale(nx, ADiag)		-- divide by diagonal
		local r = b - A(nx)
		local r2 = norm(r)
		if errorCallback and errorCallback(r2) then break end
		if r2 < epsilon then break end
		x = nx
	end
	return x
end
