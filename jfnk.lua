local table = require 'ext.table'
local gmres = require 'solver.gmres'
--[[
performs update of iteration x[n+1] = x[n] - (dF/dx)^-1 F(x[n])

args:
	f = function from #x to #x
 	x = initial vector
	dx = initial direction (optional, defaults to x)
	epsilon = (optional) tolerance to stop newton descent
	maxiter = (optional) max newton iterations
	alpha = (optional) percent of dx to trace along. default 1
	errorCallback (optional) = accepts error, iteration; returns true if iterations should be stopped
	lineSearch = (optional) line search method. options: 'none', 'linear', 'bisect'.  default 'bisect'.
	lineSearchMaxIter = (optional) iterations for the particular line search method.
	jfnkEpsilon = (optional) epsilon for derivative approximation
	gmres = gmres args
--]]
local function jfnk(args)
	local f = assert(args.f)
	local x = assert(args.x)
	local dx = args.dx or x
	local epsilon = args.epsilon or 1e-10
	local maxiter = args.maxiter or 100
	local maxAlpha = args.alpha or 1
	local errorCallback = args.errorCallback
	local lineSearch = args.lineSearch or 'bisect'
	local lineSearchMaxIter = args.lineSearchMaxIter or 100
	local jfnkEpsilon = args.jfnkEpsilon or 1e-6
	local clone = args.clone or function(v)
		return setmetatable({table.unpack(v)}, getmetatable(v))
	end
	local dot = args.dot or function(a,b)
		local n = #a
		assert(n == #b)
		local sum = 0
		for i=1,n do
			sum = sum + a[i] * b[i]
		end
		return sum
	end
	local norm = args.norm or function(x) return dot(x,x) / #x end

	local gmresArgs = args.gmres or {}	
	gmresArgs.clone = gmresArgs.clone or clone
	gmresArgs.dot = gmresArgs.dot or dot

	local function residualAtAlpha(alpha)
		return norm(f(x - dx * alpha))
	end

	local lineSearchMethods = {
		none = function() return maxAlpha end,
		linear = function()
			local bestAlpha = 0
			local bestResidual = math.huge
			for i=0,lineSearchMaxIter do
				local alpha = maxAlpha * i / lineSearchMaxIter
				local residual = residualAtAlpha(alpha)
				if residual < bestResidual then
					bestAlpha, bestResidual = alpha, residual
				end
			end
			return bestAlpha, bestResidual
		end,
		bisect = function()
			local alphaL = 0
			local alphaR = maxAlpha
			local residualL = residualAtAlpha(alphaL)
			local residualR = residualAtAlpha(alphaR)
			for i=0,lineSearchMaxIter do
				local alphaMid = .5 * (alphaL + alphaR)
				local residualMid = residualAtAlpha(alphaMid)
				if residualMid > residualL and residualMid > residualR then break end
				if residualMid < residualL and residualMid < residualR then
					if residualL <= residualR then
						alphaR, residualR  = alphaMid, residualMid
					else
						alphaL, residualL = alphaMid, residualMid
					end
				elseif residualMid < residualL then
					alphaL, residualL = alphaMid, residualMid
				else
					alphaR, residualR = alphaMid, residualMid
				end
			end
			if residualL < residualR then
				return alphaL, residualL
			else
				return alphaR, residualR
			end
		end,
	}
	local lineSearchMethod = assert(lineSearchMethods[lineSearch], "couldn't find line search method "..lineSearch)

	for iter=1,maxiter do
		local F_of_x = f(x)

		local err = norm(F_of_x)
		if errorCallback and errorCallback(err, iter) then return x end
		if err < epsilon then return x end

		-- solve dx = (dF/dx)^-1 F(x) via iterative (dF/dx) dx = f(x)
		-- use jfnk approximation for dF/dx * dx
		dx = gmres(table(gmresArgs, {
			x = dx,
			A = function(dx)
				return (f(x + dx * jfnkEpsilon) - f(x - dx * jfnkEpsilon)) / (2 * jfnkEpsilon)
			end,
			b = F_of_x,
		}))

		-- trace along dx to find minima of solution
		local alpha = lineSearchMethod()
	
		x = x - dx * alpha
	
	end

	return x
end

return jfnk
