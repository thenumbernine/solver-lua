local class = require 'ext.class'
local table = require 'ext.table'
local CLGMRes = require 'solver.cl.gmres'
local CLSolver = require 'solver.cl.solver'

local CLJFNK = class(CLSolver)

CLJFNK.needs = {
	f = true,
	scale = true,
}

--[[
performs update of iteration x[n+1] = x[n] - (dF/dx)^-1 F(x[n])

args:
	f(f_of_x, x) = function reading x and writing f_of_x 
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

	new
	dot
	norm (optiona) only used for alpha searching and error threshold testing.
		defaults to dot(x,x)
		the cpu version defaults to dot(x,x)/size
		but the gpu version never knows the size
	mulAdd
	scale
--]]
function CLJFNK:__call()
	local args = self.args

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
	
	-- how should jfnk.new and gmres.new share?
	local new = assert(args.new)
	local dot = assert(args.dot)
	local norm = args.norm or function(x) return dot(x,x) end
	local mulAdd = assert(args.mulAdd)
	local scale = assert(args.scale)
	
	local f_of_x = new'f_of_x' 
	local x_plus_dx = new'x_plus_dx'
	local x_minus_dx = new'x_minus_dx'
	local f_of_x_plus_dx = new'f_of_x_plus_dx'
	local f_of_x_minus_dx = new'f_of_x_minus_dx'

	local cache = {}
	local gmresArgs = table({
		-- allow args.gmres to overwrite these
		env = self.env,
		dot = dot,
		mulAdd = mulAdd,
		scale = scale,
		copy = args.copy,
	}, args.gmres or {}, {
		-- these overwrite args.gmres
		new = function(name)
			-- cache allocations for multiple calls
			if cache[name] then return cache[name] end
			local buffer = new(name)
			cache[name] = buffer
			return buffer
		end,
		-- this is where it helps to have access to the inplace-behavior that is wrapping jfnk-inplace
		-- for that we'd have to change from a behvaior to inheritence.  sounds good.
		x = dx,
		
		-- TODO these are accepting cl.buffers but are passing to mulAdd
		-- which expects cl.obj.buffers 
		-- ... and subsequently passes to its kernels cl.buffers ...
		-- so rather than re-wrap them
		-- instead change all solver.cl's to use cl.obj.buffers
		A = function(result, dx)
			mulAdd(x_plus_dx, x, dx, jfnkEpsilon)
			mulAdd(x_minus_dx, x, dx, -jfnkEpsilon)
			f(f_of_x_plus_dx, x_plus_dx)
			f(f_of_x_minus_dx, x_minus_dx)
			mulAdd(result, f_of_x_plus_dx, f_of_x_minus_dx, -1)
			scale(result, result, 1 / (2 * jfnkEpsilon))
		end,
		b = f_of_x,
	})
	assert(gmresArgs.A)
	local gmres = CLGMRes(gmresArgs)

	local function residualAtAlpha(alpha)
		mulAdd(x_plus_dx, x, dx, -alpha)
		f(f_of_x_plus_dx, x_plus_dx)
		return norm(f_of_x_plus_dx)
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
		f(f_of_x, x)

		local err = norm(f_of_x)
		if errorCallback and errorCallback(err, iter) then return x end
		if err < epsilon then return x end

local function buf2str(x)
	return ''
end
		-- solve dx = (dF/dx)^-1 F(x) via iterative (dF/dx) dx = f(x)
		-- use jfnk approximation for dF/dx * dx
print('solving gmres')
print('dx',buf2str(dx))
print('f_of_x',buf2str(f_of_x))
		gmres()
print('got solution dx',buf2str(dx))

		-- trace along dx to find minima of solution
		local alpha = lineSearchMethod()
	
		mulAdd(x, x, dx, -alpha)
	end

	return x
end

return CLJFNK
