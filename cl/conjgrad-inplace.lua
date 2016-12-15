--[[
This is just like the object- and operator-based ConjugateGradient.lua in the parent directory
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
local function conjGradInPlace(args)
	local A = assert(args.A)
	local b = assert(args.b)
	
	local MInv = args.MInv
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-7
	local maxiter = args.maxiter or 1000
	
	local copy = assert(args.copy)
	local new = assert(args.new)
	local free = args.free
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
	local MInvR = MInv and new() or r

	local bNorm = dot(b,b)
	if bNorm == 0 then bNorm = 1 end
	A(r, x)
	mulAdd(r, b, r, -1)
	
	if MInv then MInv(MInvR, r) end
	local rDotMInvR = dot(r, MInvR)

	repeat
		local err = dot(r, r) / bNorm
		if errorCallback and errorCallback(err, 0) then break end
		if err < epsilon then break end
		
		copy(p, MInvR)
		for iter=1,maxiter do
			A(Ap, p)
			local pDotAp = dot(p, Ap)
			local alpha = rDotMInvR / pDotAp
			mulAdd(x, x, p, alpha)
			mulAdd(r, r, Ap, -alpha)
			
			local err = dot(r, r) / bNorm
			if errorCallback and errorCallback(err, iter) then break end
			if err < epsilon then break end
			
			if MInv then MInv(MInvR, r) end
			local nRDotMInvR = dot(r, MInvR)
			local beta = nRDotMInvR / rDotMInvR
			mulAdd(p, MInvR, p, beta)
			
			rDotMInvR = nRDotMInvR
		end
	until true	-- run once, use break to jump out. my stupid CS education has scarred me from ever using goto's again.

	if free then 
		free(r) 
		free(p)
		free(Ap)
		if MInv then free(MInvR) end
	end

	return x
end

return conjGradInPlace
