local class = require 'ext.class'
local ffi = require 'ffi'

local CLSolver = class()

--[[
args:
	env = env
	size = (optional) size of buffer, in elements. default to A.domain.volume or env.base.volume
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

	-- assume our buffers are cl.obj.buffers (as newBuffer is implemented)
	-- and assume A is a cl.obj.kernel, 
	-- so wrap A to pass along the cl.buffers to the kernel
	local A = assert(args.A)
	self.args.A = function(y, x) A(y.obj, x.obj) end

	-- same with MInv
	if args.MInv then
		local MInv = args.MInv
		self.args.MInv = function(y, x) MInv(y.obj, x.obj) end
	end

	local domain = A.domain or self.env.base
	local size = self.args.size or domain.volume
	self.type = self.args.type or self.args.x.type

	self.domain = self.env:domain{size = size, dim = 1}

	-- this assumption is based on a property for Krylov solvers - that they take n iterations for a n-dimensional problem
	self.args.maxiter = self.args.maxiter or self.domain.volume

	self.args.new = self.args.new or function(...) 
		return self:newBuffer(...) 
	end

	-- hmm, this release, coupled with the __gc's release, makes things crash ...
	-- at least I know the __gc is cleaning up correctly
	--self.args.free = function(buffer) buffer.obj:release() end
	self.args.copy = self.args.copy or function(dst, src) 
		self.env.cmds:enqueueCopyBuffer{
			src = src.obj,
			dst = dst.obj,
			size = self.domain.volume * ffi.sizeof(self.type),
		}
	end

	local program = self.env:program()

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

	if not self.args.mulAdd then
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
			mulAdd(y.obj, a.obj, b.obj, ffi.new('real[1]', s))
		end
	end

	if self.needs
	and self.needs.scale
	and not self.args.scale
	then
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
			scale(y.obj, a.obj, ffi.new('real[1]', s))
		end		
	end

	program:compile()

	if not self.args.dot then
		local dot = self.env:reduce{
			size = self.domain.volume,
			op = function(x,y) return x..' + '..y end,
		}
		self.args.dot = function(a,b)
			mul(dot.buffer, a.obj, b.obj)
			return dot()
		end
	end
end

function CLSolver:newBuffer(name)
	return self.env:buffer{
		size = assert(self.domain.volume),
		type = self.type,
		name = name,
	}
end

return CLSolver
