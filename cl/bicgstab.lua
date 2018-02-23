local class = require 'ext.class'
local CLSolver = require 'solver.cl.solver'

local CLBiCGStab = class(CLSolver)

CLBiCGStab.needs = {
	A = true,
}

--[[
args:
	A = function(y,x) for x ro, y rw
		applies the linear function of A
		reads from x vector, writes to y vector
	b = object to hold 'b' vector
	x (optional) = object to hold 'x' vector.  allocates and initialized to 'b' if not provided.
	zero
	rHat (optional) =
	MInv (optional) =
	M1Inv (optional) =
	errorCallback (optional)
	epsilon (optional)
	maxiter (optional)

	new = function() creates and returns a new vector
	free = function(v) frees vector
	copy = function(dst, src) copies contents of src into dst

	dot = function(a,b) returns the inner product of a and b as a number value
	mulAdd = function(y,a,b,c) y = a + b * c, for y,a,b vectors and c scalar
--]]
function CLBiCGStab:__call()
	local args = self.args
	local A = assert(args.A)
	local b = assert(args.b)

	local MInv = args.MInv
	local M1Inv = args.M1Inv
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
		x = new'x'
		copy(x, b)
	end

	local r = new'r'

	-- r = b - A(x)
	A(r, x)
	mulAdd(r, b, r, -1)
	if MInv then MInv(r, r) end

	local rHat = args.rHat
	if not rHat then
		rHat = new'rHat'
		copy(rHat, r)
	end

	local err = dot(r,r)
	if errorCallback and errorCallback(err, 0) then return x end
	if err < epsilon then return x end

	local rho = 1
	local alpha = 1
	local omega = 1

	-- are we guaranteed that 'new' initializes to zero?
	local p = new'p'
	local v = new'v'
	local np = new'np'
	local y = new'y'
	local s = new's'
	local z = new'z'
	local t = new't'
	local M1InvT = new'M1InvT'
	
	for iter=1,maxiter do
		local nrho = dot(rHat, r)
		local beta = nrho / rho * alpha / omega
		mulAdd(np, r, p, beta)
		mulAdd(np, np, v, -omega)
		if MInv then MInv(y, np) else copy(y, np) end
		A(v, y)
		alpha = nrho / dot(rHat, v)
		mulAdd(s, r, v, -alpha)
		local s2 = dot(s,s)
		if s2 < epsilon then
			mulAdd(x, x, p, alpha)
			break
		end
		if MInv then MInv(z, s) else copy(z, s) end
		A(t, z)
		if M1Inv then M1Inv(M1InvT, t) else copy(M1InvT, t) end
		local nomega = dot(M1InvT, z) / dot(M1InvT, M1InvT)
		
		-- x := x + alpha * y + nomega * z
		mulAdd(x, x, y, alpha)
		mulAdd(x, x, z, nomega)
		
		-- TODO "if x is accurate enough then quit"
	
		-- r := s - nomega * t
		mulAdd(r, s, t, -nomega)
		
		-- TODO errorCallback
		
		local err = dot(r, r)
		if errorCallback and errorCallback(err, iter, x) then break end
		if err < epsilon then break end
	
		rho = nrho
		p, np = np, p
		omega = nomega
	end

	if free then
		free(p)
		free(v)
		free(np)
		free(y)
		free(s)
		free(z)
		free(t)
		free(M1InvT)
	end
	
	return x
end

return CLBiCGStab
