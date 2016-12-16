--[[
This is a template wrapper for an object
to setup all its OpenCL stuff
before invoking the in-place linear system solver
of its choice.

It creates and returns a class
that expects an env argument
to provide the cl.obj.env objects
--]]
local function inPlaceBehavior(inPlaceSolver)
	local class = require 'ext.class'
	local ffi = require 'ffi'

	local SolverCL = class()

	function SolverCL:init(args)
		self.env = assert(args.env)
		
		self.args = {}
		for k,v in pairs(args) do self.args[k] = v end
		
		local A = assert(args.A)
		self.args.A = function(y, x) A(y.buf, x.buf) end

		local domain = A.domain or self.env.domain
		local size = self.args.size or domain.volume
		self.type = self.args.type or self.args.x.type
	
		self.domain = require 'cl.obj.domain'{
			env = self.env,
			size = size,
			dim = 1,
		}
	
		-- this assumption is based on a property for Krylov solvers - that they take n iterations for a n-dimensional problem
		self.args.maxiter = self.args.maxiter or self.domain.volume
		
		self.args.new = self.args.new or function(...) 
			return self:newBuffer(...) 
		end

		-- hmm, this release, coupled with the __gc's release, makes things crash ...
		-- at least I know the __gc is cleaning up correctly
		--self.args.free = function(buffer) buffer.buf:release() end

		self.args.copy = self.args.copy or function(dst, src) 
			self.env.cmds:enqueueCopyBuffer{
				src = src.buf,
				dst = dst.buf,
				size = self.domain.volume * ffi.sizeof(self.type),
			}
		end

		local program = self.env:program()
	
		local mul = program:kernel{
			domain = self.domain,
			argsOut = {
				{name='y', type=self.type, buf=true},
			},
			argsIn = {
				{name='a', type=self.type, buf=true},
				{name='b', type=self.type, buf=true},
			},
			body = [[	y[index] = a[index] * b[index];]],
		}

		if not self.args.mulAdd then
			local mulAdd = program:kernel{
				domain = self.domain,
				argsOut = {
					{name='y', type=self.type, buf=true},
				},
				argsIn = {
					{name='a', type=self.type, buf=true},
					{name='b', type=self.type, buf=true},
					{name='s', type='real'},
				},
				body = [[	y[index] = a[index] + b[index] * s;]],
			}
			self.args.mulAdd = function(y,a,b,s)
				mulAdd(y.buf, a.buf, b.buf, ffi.new('real[1]', s))
			end
		end

		program:compile()

		if not self.args.dot then
			local dotBuf = self.env:buffer{
				name = 'dotBuf',
				size = self.domain.volume,
				type = self.type,
			}
	
			local dot = self.env:reduce{
				size = self.domain.volume,
				buffer = dotBuf.buf,
				op = function(x,y) return x..' + '..y end,
			}
			self.args.dot = function(a,b)
				mul(dotBuf.buf, a.buf, b.buf)
				return dot()
			end
		end
	end

	function SolverCL:newBuffer(name)
		return self.env:buffer{
			size = self.domain.volume,
			type = self.type,
		}
	end

	function SolverCL:__call()
		inPlaceSolver(self.args)
	end

	return SolverCL
end

return inPlaceBehavior
