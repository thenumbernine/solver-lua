--[[
This is a template wrapper for an object
to setup all its OpenCL stuff
before invoking the in-place linear system solver
of its choice.

It creates and returns a class
that expects an env argument
to provide the cl.obj.env objects
--]]
local function clInPlaceBehavior(inPlaceSolver)
	local class = require 'ext.class'
	local ffi = require 'ffi'

	local SolverCL = class()

	function SolverCL:init(args)
		self.env = assert(args.env)
		
		self.args = {}
		for k,v in pairs(args) do self.args[k] = v end
		
		local A = assert(args.A)
		self.args.A = function(y, x) A(y.buf, x.buf) end

		-- this assumption is based on a property for Krylov solvers - that they take n iterations for a n-dimensional problem
		self.args.maxiter = self.args.maxiter or self.env.volume

		self.args.new = self.args.new or function() return self.env:buffer() end

		-- hmm, this release, coupled with the __gc's release, makes things crash ...
		-- at least I know the __gc is cleaning up correctly
		--self.args.free = function(buffer) buffer.buf:release() end

		self.args.copy = self.args.copy or function(dst, src) 
			self.env.cmds:enqueueCopyBuffer{
				src = src.buf,
				dst = dst.buf,
				-- TODO? store the size in the cl.buffer?  or the cl.obj.buffer?
				size = self.env.volume * ffi.sizeof(dst.type),
			}
		end

		local program = self.env:program()

		local dotBuf = self.env:buffer{name='dotBuf'}
		local mul = program:kernel{
			argsOut = {{name='y', buf=true}},
			argsIn = {{name='a', buf=true}, {name='b', buf=true}},
			body = [[	y[index] = a[index] * b[index];]],
		}

		if not self.args.mulAdd then
			local mulAdd = program:kernel{
				argsOut = {{name='y', buf=true}},
				argsIn = {{name='a', buf=true}, {name='b', buf=true}, {name='s', type='real'}},
				body = [[	y[index] = a[index] + b[index] * s;]],
			}
			self.args.mulAdd = function(y,a,b,s)
				mulAdd(y.buf, a.buf, b.buf, ffi.new('real[1]', s))
			end
		end

		program:compile()

		if not self.args.dot then
			local dot = self.env:reduce{
				buffer = dotBuf.buf,
				op = function(x,y) return x..' + '..y end,
			}
			self.args.dot = function(a,b)
				mul(dotBuf.buf, a.buf, b.buf)
				return dot()
			end
		end
	end

	function SolverCL:__call()
		inPlaceSolver(self.args)
	end

	return SolverCL
end

return clInPlaceBehavior
