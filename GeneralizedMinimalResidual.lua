local table = require 'ext.table'

local backSubstituteUpperTriangular = require 'LinearSolvers.backSubstituteUpperTriangular'

local function updateX(x, h, s, v, i)
	--local y = h(1:i,1:i) \ s(1:i)		-- and exit
	local subS = {table.unpack(s,1,i)}
	local subH = {}
	for j=1,i do
		subH[j] = {table.unpack(h[j], 1, i)}
	end
	local y = backSubstituteUpperTriangular(subH, subS)
	--x = x + V(:,1:i)*y
	for j=1,i do
		x = x + v[j] * y[j]
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

--[[
source: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
		http://www.netlib.org/templates/matlab/gmres.m
args:
	A = linear function A : x -> x
	b = solution vector
	x0 (optional) = initial guess vector
	clone = vector clone function
	dot = vector dot function
	norm = (optional) vector norm. deault L2 norm via dot()
	MInv = (optional) inverse of preconditioner linear function MInv : x -> x
	errorCallback (optional) = function(|r|/|b|, iteration, x, |r|^2, |b|^2)
		accepts error, iteration; returns true if iterations should be stopped
	epsilon (optional) = error threshold at which to stop
	maxiter (optional) = maximum iterations to run
	restart (optional) = maximum iterations between restarts

vectors need operators + - scalar* scalar/ # []

--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local norm = args.norm or function(x) return math.sqrt(dot(x,x)) end
	local MInv = args.MInv or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-20
	local maxiter = args.maxiter or 100
	local m = args.restart or #args.b

	local bLen = norm(b)
	
	local x = clone(args.x0 or b)
	local r = MInv(b - A(x))
	local rLen = norm(r)

	local err = bLen > 0 and rLen / bLen or rLen
	if errorCallback and errorCallback(err, 0, x, rLen*rLen, bLen*bLen) then return x end
	if err < epsilon then return x end

	local v = {}	--v[m+1][n]
	for i=1,m+1 do
		v[i] = {}
		for j=1,#b do
--			v[i][j] = 0
		end
	end

	local h = {}	--h[m+1][m]
	for i=1,m+1 do
		h[i] = {}
		for j=1,m do
			h[i][j] = 0
		end
	end

	local cs = {}	--cs[m]
	for i=1,m do
		cs[i] = 0
	end
	
	local sn = {}	--sn[m]
	for i=1,m do
		sn[i] = 0
	end

	local s = {}	--s[m+1]
	
	local iter = 0
	while true do

		v[1] = r/rLen
		for i=2,m+1 do
			s[i] = 0
		end
		s[1] = rLen

		for i=1,m do		-- construct orthonormal basis using Gram-Schmidt
			iter = iter + 1
			if iter >= maxiter then break end
			
			local w = MInv(A(v[i]))
			for k=1,i do
				h[k][i] = dot(w, v[k])
				w = w - v[k] * h[k][i]
			end
			h[i+1][i] = norm(w)
			v[i+1] = w / h[i+1][i]
			for k=1,i-1 do	-- apply Givens rotation
				h[k][i], h[k+1][i] =
					cs[k] * h[k][i] + sn[k] * h[k+1][i],
					-sn[k] * h[k][i] + cs[k] * h[k+1][i]
			end
			cs[i], sn[i] = rotmat(h[i][i], h[i+1][i]) -- form i-th rotation matrix
			s[i], s[i+1] = cs[i] * s[i], -sn[i] * s[i]	-- approximate residual norm
			h[i][i] = cs[i] * h[i][i] + sn[i] * h[i+1][i]
			h[i+1][i] = 0
			
			local err = math.abs(bLen > 0 and s[i+1] / bLen or s[i+1])
			if errorCallback and errorCallback(err, iter, x, err*err*bLen*bLen, bLen*bLen) then return x end
			if err < epsilon then	-- update approximation
				x = updateX(x, h, s, v, i)
				return x
			end
		end
		if iter >= maxiter then break end
		
		x = updateX(x, h, s, v, m)
	
		r = MInv(b - A(x))		-- compute residual
		rLen = norm(r)
		s[m+1] = rLen
		local err = bLen > 0 and rLen / bLen or rLen		-- check convergence
		if err < epsilon then break end
	end
	return x
end
