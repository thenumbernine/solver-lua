local class = require 'ext.class'
local table = require 'ext.table'
local range = require 'ext.range'

-- hmm, I should gpu-ize this too ...
local backsub = require 'solver.backsub'

-- x = CL buffer
-- h = [m+1][m] Lua table
-- s = [m+1] Lua table
-- v = [m+1] CL buffers
-- i = index in s, h, etc
local function updateX(x, h, s, v, i, mulAdd)
	--local y = h(1:i,1:i) \ s(1:i)		-- and exit
	local subS = {table.unpack(s,1,i)}
	local subH = {}
	for j=1,i do
		subH[j] = {table.unpack(h[j], 1, i)}
	end
	local y = backsub(subH, subS)
	--x = x + V(:,1:i)*y
	for j=1,i do
		mulAdd(x, x, v[j], y[j])	
	end
	return x
end

-- http://www.mathworks.com/matlabcentral/fileexchange/2158-templates-for-the-solution-of-linear-systems/content/templates/rotmat.m
local function rotmat(a,b)
	if b == 0 then
		return 1, 0
	elseif math.abs(b) > math.abs(a) then
		local temp = a / b
		local s = 1 / math.sqrt(1 + temp * temp)
		return temp * s, s
	else
		local temp = b / a
		local c = 1 / math.sqrt(1 + temp * temp)
		return c, temp * c
	end
end

local GMResInPlace = class()

-- tell the inplace-behavior we need a scale kernel
GMResInPlace.needs = {
	scale = true,
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
	x (optional) = object to hold 'x' vector.  initialized to 'b' if not provided.
	MInv = (optional) function(y,x) for x ro and y rw vectors. preconditioner
	errorCallback (optional) = function(|r|/|b|, iteration, x, |r|^2, |b|^2)
		returns true if iterations should be stopped
	epsilon (optional)
	maxiter (optional)
	restart (optional) = maximum iterations between restarts

	new = function() returns and create a new vector
	free = function(v) frees vector
	copy = function(dst, src) copies contents of src into dst

	dot = function(a,b) returns a number of the inner product of a and b
	mulAdd = function(y,a,b,c) y = a + b * c, for y,a,b vectors and c scalar
	scale = function(y,x,s) y = x * s, for y,x vectors and s scalar
--]]
function GMResInPlace:__call(args)
	local A = assert(args.A)
	local b = assert(args.b)

	local MInv = args.MInv
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-7
	local maxiter = args.maxiter or 1000
	local m = args.restart or 10	-- don't default this to anything too big

	local copy = assert(args.copy)
	local new = assert(args.new)
	local free = args.new
	local dot = assert(args.dot)
	local mulAdd = assert(args.mulAdd)
	local scale = assert(args.scale)

	local x = args.x
	if not x then
		x = new'x'
		copy(x, b)
	end

	local r = new'r'	--[n]
	
	local bLen = math.sqrt(dot(b,b))

	-- r = M^-1 (b - A x)
	A(r, x)
	mulAdd(r, b, r, -1)
	if MInv then MInv(r, r) end

	repeat	-- runs only once.  used for break.
		local rLen = math.sqrt(dot(r, r))
		local err = rLen / bLen
		if errorCallback and errorCallback(err, 0, x, rLen*rLen, bLen*bLen) then break end
		if err < epsilon then break end

		-- all these initialize to zero
		local w = new'w'	--[n]
		-- choosing a big restart is dangerous ...
		local v = range(m+1):map(function(i)
			return new('v'..i)	--[m+1][n]
		end)
		-- if restart is gonna be small then maybe these should be Lua tables:
		local h = range(m+1):map(function(i)	--[m+1][m]
			return range(m):map(function(i)
				return 0
			end) 
		end)
		local cs = {}	--[m]
		for i=1,m do cs[i] = 0 end
		local sn = {}	--[m]
		for i=1,m do sn[i] = 0 end
		local s = {}	--[m+1]

		local iter = 0
		while true do

			scale(v[1], r, 1/rLen)
		
			-- s = [rLen, 0, 0, ...]
			s[1] = rLen
			for i=2,m+1 do
				s[i] = 0
			end
	
			for i=1,m do		-- construct orthonormal basis using Gram-Schmidt
				iter = iter + 1
				if iter >= maxiter then break end

				-- w = M^-1 A v
				A(w, v[i])
				if MInv then MInv(w, w) end

				for k=1,i do
					h[k][i] = dot(w, v[k])
					mulAdd(w, w, v[k], -h[k][i])
				end
				h[i+1][i] = math.sqrt(dot(w,w))
				scale(v[i+1], w, 1/h[i+1][i])
				for k=1,i-1 do	-- apply Givens rotation
					h[k][i], h[k+1][i] =
						cs[k] * h[k][i] + sn[k] * h[k+1][i],
						-sn[k] * h[k][i] + cs[k] * h[k+1][i]
				end
				cs[i], sn[i] = rotmat(h[i][i], h[i+1][i]) -- form i-th rotation matrix
				s[i], s[i+1] = cs[i] * s[i], -sn[i] * s[i]	-- approximate residual norm
				h[i][i] = cs[i] * h[i][i] + sn[i] * h[i+1][i]
				h[i+1][i] = 0

				local err = math.abs(s[i+1]) / (bLen > 0 and bLen or 1)
				if errorCallback and errorCallback(err, iter, x, err*err*bLen*bLen, bLen*bLen) then return x end
				if err < epsilon then	-- update approximation
					updateX(x, h, s, v, i, mulAdd)
					return x
				end
			end
			if iter >= maxiter then break end
		
			updateX(x, h, s, v, m, mulAdd)

			-- r = M^-1 (b - A x)
			A(r, x)
			mulAdd(r, b, r, -1)
			if MInv then MInv(r, r) end
			
			rLen = math.sqrt(dot(r,r))
			s[m+1] = rLen
			local err = rLen / (bLen > 0 and bLen or 1)		-- check convergence
			if err < epsilon then break end
		end

		if free then
			free(w)
			free(v)
			free(h)
			free(cs)
			free(sn)
			free(s)
		end
	
	until true -- just run once / use for break jumps

	if free then
		free(r)
	end

	return x
end

return GMResInPlace()