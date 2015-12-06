local HouseholderQR = require 'LinearSolvers.HouseholderQR'

-- solves for x in the system y = u x for upper triangular matrix u
local function backSubstituteUpperTriangular(u, y)
	local n = #u
	local x = {}
	for i=n,1,-1 do
		--[[
		y[n] = x[n] * u[n][n]
		y[n-1] = x[n-1] * u[n-1][n-1] + x[n] * u[n-1][n]
		y[n-2] = x[n-2] * u[n-2][n-2] + x[n-1] * u[n-2][n-1] + x[n] * u[n-2][n]
		y[i] = sum[j=i to n] x[j] * u[i][j]
		y[i] = x[i] * u[i][i] + sum[j=i+1 to n] x[j] * u[i][j]
		x[i] = (y[i] - sum[j=i+1 to n] x[j] * u[i][j]) / u[i][i]
		--]]
		local sum = 0
		for j=i+1,n do
			sum = sum + x[j] * u[i][j]
		end
		x[i] = y[i] - sum / u[i][i]
	end
	return x
end

local function updateX(x, H, s, Vt, i)
	--local y = H(1:i,1:i) \ s(1:i)		-- and exit
	local subH = {}
	local subS = {}
	for j=1,i do
		subS[j] = s[j]
		subH[j] = {}
		for k=1,i do
			subH[j][k] = H[j][k]
		end
	end
	local q,r = HouseholderQR(subH, subS)	
	-- H y = s
	-- qr y = s
	-- r y = q^-1 s = q^t s
	local qTsubS = {}
	for j=1,i do
		local sum = 0
		for k=1,i do
			sum = sum + q[k][j] * s[k]
		end
		qTsubS[j] = sum
	end
	local y = backSubstituteUpperTriangular(r, qTsubS)

	--x = x + V(:,1:i)*y
	for j=1,i do
		x = x + Vt[j] * y[j]
	end
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
	MInv = inverse of preconditioner linear function MInv : x -> x
	errorCallback (optional) = accepts error, iteration; returns true if iterations should be stopped
	epsilon (optional) = error threshold at which to stop
	maxiter (optional) = maximum iterations to run
	restart (optional) = maximum iterations between restarts

vectors need operators + - * / []

--]]
return function(args)
	local A = assert(args.A)
	local b = assert(args.b)
	local clone = assert(args.clone)
	local dot = assert(args.dot)
	local MInv = args.MInv or clone
	local errorCallback = args.errorCallback
	local epsilon = args.epsilon or 1e-50
	local maxiter = args.maxiter or 100
	local restart = args.restart or 100

	local function norm(x) return dot(x,x) end

	local iter = 0		-- initialization
	local flag = 0

	local bnrm2 = norm(b);
	if bnrm2 == 0 then bnrm2 = 1 end

	local x = clone(args.x0 or b)
	local r = MInv(b - A(x))
	local err = norm(r) / bnrm2
	if errorCallback and errorCallback(err, 0) then return x end
	if err < epsilon then return end

	local n = #x		-- initialize workspace
	local m = restart

	local Vt = {}
	for i=1,m+1 do
		Vt[i] = clone(x)
		for j=1,n do
			Vt[i][j] = 0
		end
	end

	local H = {}
	for i=1,m+1 do
		H[i] = {}
		for j=1,m do
			H[i][j] = 0
		end
	end
	
	local cs = {}
	for i=1,m do
		cs[i] = 0
	end
	
	local sn = {}
	for i=1,m do
		sn[i] = 0
	end

	local e1 = clone(x)
	e1[1] = 1
	for i=2,n do
		e1[i] = 0
	end

	for iter=1,maxiter do		-- begin iteration
		r = MInv(b - A(x))
		Vt[i] = r / norm(r)
		local s = norm(r) * e1
		for i=1,m do		-- construct orthonormal
			local w = clone(x)
			for j=1,n do
				w[j] = MInv(A(Vt[i]))		-- basis using Gram-Schmidt
			end
			for k=1,i do
				H[k][i] = dot(w, Vt[k])
				w = w - H[k][i] * Vt[k]
			end
			H[i+1][i] = norm(w)
			Vt[i+1] = w / H[i+1][i]
			for k=1,i-1 do	-- apply Givens rotation
				local temp = cs[k] * H[k][i] + sn[k] * H[k+1][i]
				H[k+1][i] = -sn[k] * H[k][i] + cs[k] * H[k+1][i]
				H[k][i]	= temp
			end
			cs[i], sn[i] = rotmat(H[i][i], H[i+1][i]) -- form i-th rotation matrix
			temp = cs[i] * s[i]		-- approximate residual norm
			s[i+1] = -sn[i] * s[i]
			s[i] = temp
			H[i][i] = cs[i] * H[i][i] + sn[i] * H[i+1][i]
			H[i+1][i] = 0
			
			local err = abs(s[i+1]) / bnrm2
			if errorCallback and errorCallback(err, iter) then return x end
			if err < epsilon then	-- update approximation
				updateX(x, H, s, Vt, i)
				return x
			end
		end
		updateX(x, H, s, Vt, m)
	
		r = MInv(b - A(x))		-- compute residual
		s[i+1] = norm(r)
		local err = s[i+1] / bnrm2		-- check convergence
		if err < epsilon then return x end
	end
end
