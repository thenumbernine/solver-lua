local math = require 'ext.math'	-- isfinite
local class = require 'ext.class'
local CLSolver = require 'solver.cl.solver'

local CLConjRes = class(CLSolver)

CLConjRes.needs = {
	A = true,
}

--[[
This is just like the object- and operator-based ConjugateResidual.lua in the parent directory
except it has a focus on reuing objects.
This is for the sake of using it with GPU acceleration.

args:
	A = function(y,x) for x ro, y rw
		applies the linear function of A
		reads from x vector, writes to y vector
	b = object to hold 'b' vector
	x (optional) = object to hold 'x' vector.  allocates and initialized to 'b' if not provided.
	MInv (optional) = function(y,x) for x ro and y rw vectors. preconditioner
	errorCallback (optional) = function(|r|/|b|, iteration, x, |r|^2, |b|^2)
		returns true if iterations should be stopped
	epsilon (optional)
	maxiter (optional)
	
	new = function() creates and returns a new vector
	free = function(v) frees vector
	copy = function(dst, src) copies contents of src into dst

	dot = function(a,b) returns the inner product of a and b as a number value
	mulAdd = function(y,a,b,c) y = a + b * c, for y,a,b vectors and c scalar
--]]
function CLConjRes:__call()
	local args = self.args
	local A = assert(args.A)	-- A : n -> m
	local b = assert(args.b)	-- m

	local MInv = args.MInv		-- MInv : m -> n
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-7
	local maxiter = args.maxiter or 1000

	local copy = assert(args.copy)
	local new = assert(args.new)
	local free = args.free
	local dot = assert(args.dot)
	local mulAdd = assert(args.mulAdd)

	local x = args.x			-- n
	if not x then
		x = new'x'
		copy(x, b)
	end

	local r = new'r'			-- m
	local p = new'p'			-- n
	local Ap = new'Ap'			-- m
	local Ar = new'Ar'
	local MInvAp = MInv and new'MInvAp' or Ap

	local bSq = dot(b,b)		-- m, m -> 1
	if not math.isfinite(bSq) then return false, "|b| is not finite" end

	A(r, x)						-- A(x) : m  
	mulAdd(r, b, r, -1)			-- r : m
	if MInv then MInv(r, r) end	-- MInv(r) : n

	repeat
		local rSq = dot(r, r)
		local err = math.sqrt(rSq / (bSq > 0 and bSq or 1))
		if errorCallback and errorCallback(err, 0, x, rSq, bSq) then break end
		if not math.isfinite(err) then return false, "r dot r is not finite" end
		if err < epsilon then break end

		-- r should be of dim n here ... which it is at least a subset of with m >= n.
		-- Well this is pretty clear that r and x need to be the same dim.
		A(Ar, r)				-- A(r) : m
		local rAr = dot(r, Ar)	-- m, m -> 1
		if not math.isfinite(rAr) then return false, "r dot A(r) is not finite" end
		copy(p, r)
		A(Ap, p)
		for iter=1,maxiter do
			if MInv then MInv(MInvAp, Ap) end
			local ApDotMInvAp = dot(Ap, MInvAp)
			if ApDotMInvAp == 0 then return false, "A(p) dot M^-1(A(p)) == 0" end 
			local alpha = rAr / ApDotMInvAp
			if not math.isfinite(alpha) then return false, "alpha is not finite" end
			mulAdd(x, x, p, alpha)
			mulAdd(r, r, MInvAp, -alpha)
	
			rSq = dot(r, r)
			local err = math.sqrt(rSq / (bSq > 0 and bSq or 1))
			if errorCallback and errorCallback(err, iter, x, rSq, bSq) then break end
			if not math.isfinite(err) then return false, "error is not finite" end
			if err < epsilon then break end
		
			A(Ar, r)
			local nrAr = dot(r, Ar)
			if not math.isfinite(nrAr) then return false, "next r dot A(r) is not finite" end
			if rAr == 0 then return false, "r dot A(r) == 0" end
			local beta = nrAr / rAr
			if not math.isfinite(beta) then return false, "beta is not finite" end

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

return CLConjRes
