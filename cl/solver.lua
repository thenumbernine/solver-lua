local class = require 'ext.class'
local ffi = require 'ffi'

local CLSolver = class()

--[[
args:
	env = env
	count = (optional) number of elements in the buffer. default to A.domain.volume or env.base.volume.
	type = ctype of buffer
	maxiter = maxiter of solver
	new = (optional) function(name) = allocator
	copy = (optional) function(dst, src) copy function
		dst, src are of type cl.obj.buffer 
	mulAdd = (optional) function(y, a, b, s) performs y = a + b * s
		y,a,b are cl.obj.buffer, s = number
	
	only in gmres, jfnk:
	scale = (optional) function(y, a, s) performs y = a * s
		y,a are cl.obj.buffer, s = number

	only in jfnk:
	f = function(y,x) performs the function to minimize
		y,x are cl.obj.buffer

	only in conjgrad, conjres, gmres:
	MInv = function(y,x) preconditioner
		x and y are of type cl.obj.buffer
	A = function(y, x) that reads x and stores the function result in 'y' 
		x and y are of type cl.obj.buffer	
	...
	the rest of the args are passed on to 'inPlaceSolver'
--]]
function CLSolver:init(args)
	self.env = assert(args.env)
	
	self.args = {}
	for k,v in pairs(args) do self.args[k] = v end

	-- only in conjgrad, conjres, gmres:
	-- assume our buffers are cl.obj.buffers (as newBuffer is implemented)
	if self.needs.A then
		self.args.A = assert(args.A)
	end

	-- only in jfnk:
	if self.needs.f then
		self.args.f = assert(args.f)
	end

	-- same with MInv
	if args.MInv then
		self.args.MInv = args.MInv
	end

	-- how to determine the domain?
	local domain = 
		args.domain
		or (args.A and type(args.A) == 'table' and args.A.domain)
		or (args.f and type(args.f) == 'table' and args.f.domain)
		or self.env.base
	local count = self.args.count or domain.volume
	self.type = self.args.type or (self.args.x and self.args.x.type) or 'real'

	self.domain = self.env:domain{size = count, dim = 1}

	-- this assumption is based on a property for Krylov solvers - that they take n iterations for a n-dimensional problem
	self.args.maxiter = self.args.maxiter or self.domain.volume

	self.args.new = self.args.new or function(...) 
		return self:newBuffer(...) 
	end

	-- hmm, this release, coupled with the __gc's release, makes things crash ...
	-- at least I know the __gc is cleaning up correctly
	--self.args.free = function(buffer) buffer.obj:release() end
	self.args.copy = self.args.copy or function(dst, src) 
		self.env.cmds[1]:enqueueCopyBuffer{
			src = src.obj,
			dst = dst.obj,
			size = self.domain.volume * ffi.sizeof(self.type),
		}
	end

	local program
	local function makeProgram()
		program = program or self.env:program()
	end
	if not self.args.mulAdd then
		makeProgram()
		local mulAdd = program:kernel{
			domain = self.domain,
			argsOut = {
				{name='y', type=self.type, obj=true},
			},
			argsIn = {
				{name='a', type=self.type, obj=true},
				{name='b', type=self.type, obj=true},
				{name='s', type='real'},
			},
			body = [[	y[index] = a[index] + b[index] * s;]],
		}
		self.args.mulAdd = function(y,a,b,s)
			mulAdd(y, a, b, ffi.new('real[1]', s))
		end
	end

	if self.needs
	and self.needs.scale
	and not self.args.scale
	then
		makeProgram()
		local scale = program:kernel{
			domain = self.domain,
			argsOut = {
				{name='y', type=self.type, obj=true},
			},
			argsIn = {
				{name='a', type=self.type, obj=true},
				{name='s', type='real'},
			},
			body = [[	y[index] = a[index] * s;]],
		}
		self.args.scale = function(y,a,s)
			scale(y, a, ffi.new('real[1]', s))
		end		
	end

	if not self.args.dot then
		makeProgram()
		local mul = program:kernel{
			domain = self.domain,
			argsOut = {
				{name='y', type=self.type, obj=true},
			},
			argsIn = {
				{name='a', type=self.type, obj=true},
				{name='b', type=self.type, obj=true},
			},
			body = [[	y[index] = a[index] * b[index];]],
		}
	
		local dot = self.env:reduce{
			count = self.domain.volume,
			op = function(x,y) return x..' + '..y end,
		}
		self.args.dot = function(a,b)
			mul(dot.buffer, a, b)
			return dot()
		end
	end

	if program then
		program:compile()
	end
end

function CLSolver:newBuffer(name)
	return self.domain:buffer{
		type = self.type,
		name = name,
	}
end

return CLSolver
