local math = require 'ext.math'	-- isfinite

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

This currently only works with square conjugate gradient solutions.

To make it work for rectangular A : R^m -> R^n, m >= n,
(m is output dimension, n is input dimension)
you would need to take note:
	- copy() copies n elements
	- dot() maps R^m * R^n -> 1
	- A maps R^n -> R^m, MInv map R^m -> R^n
	- new() will allocate n for 'x', 'p', 'MInvR', and m for 'r', 'Ap'

for m < n, r would be of dim m, so you couldn't eliminate all n nullspace dimensions,
so I wouldn't imagine you could guarantee a solution.
--]]
local function conjGradInPlace(args)
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
		copy(x, b)				-- n
	end
	
	local r = new'r'			-- m
	local p = new'p'			-- n
	local Ap = new'Ap'			-- m
	
	-- if MInv is omitted then r is used directly for MInvR, since no computation is necessary.
	-- if our problem is rectangular with m >= n then the MInvR operation can be thought of as truncation
	--  from r's m elements to MInvR's n elements.
	local MInvR = MInv and new'MInvR' or r	-- n

	-- here's a place where the dot operates on m, m instead of m, n
	-- but for m >= n we still wouldn't crash, the dot would just truncate the data
	local bNorm = dot(b,b)		-- b dot b : m, m -> 1.  
	if bNorm == 0 then bNorm = 1 end
	A(r, x)						-- A(x) : m
	mulAdd(r, b, r, -1)			-- r : m
	
	if MInv then MInv(MInvR, r) end	-- MInv(r) : n
	local rDotMInvR = dot(r, MInvR)	-- r dot MInv(r) : m, n -> 1

	repeat
		local err = dot(r, r) / bNorm
		if errorCallback and errorCallback(err, 0, x) then break end
		if not math.isfinite(err) then return false, "error is not finite" end
		if err < epsilon then break end
		
		copy(p, MInvR)					-- p : n
		for iter=1,maxiter do
			A(Ap, p)							-- Ap : m
			local ApDotP = dot(Ap, p)			-- A(p) dot p : m, n -> 1
			local alpha = rDotMInvR / ApDotP
			mulAdd(x, x, p, alpha)				-- x : n
			mulAdd(r, r, Ap, -alpha)			-- r : m
			
			if MInv then MInv(MInvR, r) end		-- MInv(r) : n
			local nRDotMInvR = dot(r, MInvR)	-- r dot MInv(r) : m, n -> 1
			
			local err = nRDotMInvR / bNorm
			if errorCallback and errorCallback(err, iter, x) then break end
			if not math.isfinite(err) then return false, "error is not finite" end
			if err < epsilon then break end
			
			local beta = nRDotMInvR / rDotMInvR
			rDotMInvR = nRDotMInvR
			
			mulAdd(p, MInvR, p, beta)			-- p : n
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
