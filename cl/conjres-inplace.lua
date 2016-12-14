--[[
This is just like the object- and operator-based ConjugateResidual.lua in the parent directory
except it has a focus on reuing objects.
This is for the sake of using it with GPU acceleration.

args:
	A = function(y,x) for x ro, y rw
		applies the linear function of A
		reads from x vector, writes to y vector
	b = object to hold 'b' vector
	x (optional) = object to hold 'x' vector.  initialized to 'b' if not provided.
	MInv = (optional) function(y,x) for x ro and y rw vectors. preconditioner
	errorCallback (optional) returns true to stop
	epsilon (optional)
	maxiter (optional)
	
	new = function() returns and create a new vector
	free = function(v) frees vector
	copy = function(dst, src) copies contents of src into dst

	dot = function(a,b) returns a number of the inner product of a and b
	mulAdd = function(y,a,b,c) y = a + b * c, for y,a,b vectors and c scalar
--]]
local function conjResInPlace(args)
	local A = assert(args.A)
	local b = assert(args.b)

	local MInv = args.MInv
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-7
	local maxiter = args.maxiter or 1000

	local copy = assert(args.copy)
	local new = assert(args.new)
	local free = args.new
	local dot = assert(args.dot)
	local mulAdd = assert(args.mulAdd)

	local x = args.x
	if not x then
		x = new()
		copy(x, b)
	end

	local r = new()
	local p = new()
	local Ap = new()
	local Ar = new()
	local MInvAp = MInv and new() or Ap

	local bNorm = dot(b,b)
	if bNorm == 0 then bNorm = 1 end

	A(r, x)
	mulAdd(r, b, r, -1)
	if MInv then MInv(r, r) end

	repeat
		local err = dot(r, r) / bNorm
		if errorCallback and errorCallback(err, 0) then break end
		if err < epsilon then break end

		A(Ar, r)
		local rAr = dot(r, Ar)
		copy(p, r)
		A(Ap, p)
		for iter=1,maxiter do
			if MInv then MInv(MInvAp, Ap) end
			local alpha = rAr / dot(Ap, MInvAp)
			mulAdd(x, x, p, alpha)
			mulAdd(r, r, MInvAp, -alpha)
		
			local err = dot(r, r) / bNorm
			if errorCallback and errorCallback(err, iter) then break end
			if err < epsilon then break end
		
			A(Ar, r)
			local nrAr = dot(r, Ar)
			local beta = nrAr / rAr

			rAr = nrAr
			mulAdd(p, r, p, beta)
			mulAdd(Ap, Ar, Ap, beta)
		end
	
	until true -- just run once / use for break jumps

	if free then
		free(r)
		free(p)
		free(Ap)
		free(Ar)
		if MInv then free(MInvAp) end
	end

	return x
end

return conjResInPlace
