#!/usr/bin/env luajit
require 'ext'
--[[
trying to do multi-gpu with separate buffers and update boundary regions
--]]
local cl = require 'ffi.OpenCL'
local matrix = require 'matrix'
local gnuplot = require 'gnuplot'
local ffi = require 'ffi'
local class = require 'ext.class'

local n = 128

local env = require 'cl.obj.env'{
	size = {n, n},
	precision = 'float',
	queue = {
		properties = cl.CL_QUEUE_PROFILING_ENABLE,
	},
}

local regionType = [[
typedef struct region_t {
	int4 supersize;	//size of the parent buffer
	int4 size;		//size of this buffer
	int4 step;		//step of this buffer (based on subbuffer / xreadmax-xreadmin)

	//(n-1)-axis chopped up
	int readmin;	//sub-buffer range
	int readmax;
	int writemin;	//kernel range
	int writemax;
} region_t;
]]
ffi.cdef(regionType)


local domains = table()
local regions = table()		-- CPU
local regionBufs = table()	-- GPU

for i,device in ipairs(env.devices) do
	local nymin = math.floor(n*(i-1)/#env.devices)
	local nymax = math.floor(n*i/#env.devices)

	local region = ffi.new'region_t[1]'
	region[0].supersize.x = n
	region[0].supersize.y = n
	region[0].supersize.z = 1

	region[0].writemin = nymin
	region[0].writemax = nymax

	if i > 1 then nymin = nymin - 1 end
	if i < #env.devices then nymax = nymax + 1 end

	region[0].readmin = nymin
	region[0].readmax = nymax

	region[0].size = region[0].supersize
	region[0].size.y = region[0].readmax - region[0].readmin

	region[0].step.x = 1
	region[0].step.y = region[0].size.x
	region[0].step.z = region[0].size.y * region[0].step.x
	region[0].step.w = region[0].size.z * region[0].step.y

	domains:insert(require 'cl.obj.domain'{
		env = env,
		device = device,
		size = {region[0].size.x, region[0].size.y},
	})

	local regionBuf = env:buffer{
		name = 'region',
		type = 'region_t',
		count = 1,
		readwrite = 'read',
		constant = true,
		data = region,
	}
	regions:insert(region)
	regionBufs:insert(regionBuf)
end

-- holds chopped up buffers spread across several devices
-- also abstracts anything that the solver is going to do with the buffer
local MultiDeviceBuffer = class()

--[[ TODO this looks incomplete
function MultiDeviceBuffer:init()
	self.bufs = table()
	for i,region in ipairs(regions) do
		self.buf = env:buffer
	end
end
--]]

-- A x = b ... solve for x
local b = MultiDeviceBuffer()

-- TODO returning 'x' doesn't return an env:buffer object
local x = MultiDeviceBuffer()

local program = env:program{
	code = table{
		regionType,
		[[
kernel void bInit(
	constant region_t* region,
	global float* b
) {
	//cl global_id range / range in write
	int4 iwr = globalInt4();
	if (iwr.x >= region->size.x) return;
	if (iwr.y >= region->writemax - region->writemin) return;

	//range in subbuffer
	int4 ird = iwr;
	ird.y += region->readmin - region->writemin;
	int subindex = indexForInt4ForSize(ird, region->size.x, region->size.y, region->size.z);

	//range in the super buffer
	int4 i = iwr;
	i.y += region->writemin;

	b[subindex] = (
		i.x >= region->supersize.x/4 && i.x < region->supersize.x*3/4 &&
		i.y >= region->supersize.y/4 && i.y < region->supersize.y*3/4
	) ? 1. : 0.;
}

kernel void A(
	constant region_t* region,
	global float* y,
	global const float* x
) {
	//cl global_id range / range in write
	int4 iwr = globalInt4();
	if (iwr.x >= region->size.x) return;
	if (iwr.y >= region->writemax - region->writemin) return;

	//range in subbuffer
	int4 ird = iwr;
	ird.y += region->readmin - region->writemin;
	int subindex = indexForInt4ForSize(ird, region->size.x, region->size.y, region->size.z);

	//range in the super buffer
	int4 i = iwr;
	i.y += region->writemin;

	//boundary
	if (i.x == 0 ||
		i.y == 0 ||
		i.x == region->supersize.x-1 ||
		i.y == region->supersize.y-1)
	{
		y[subindex] = x[subindex];
		return;
	}

	//PDE
	const real hSq = .01;
	y[subindex] = -(
		x[subindex + region->step.x]
		+ x[subindex - region->step.x]
		+ x[subindex + region->step.y]
		+ x[subindex - region->step.y]
		- 4. * x[subindex]
	) / hSq;
}

]],
	}:concat'\n'
}
program:compile()


local function chopUpBuffer(buf)
	local bufs = table()
	for i,device in ipairs(env.devices) do
		local region = regions[i]
		local start = region[0].step.y * region[0].readmin
		local fin = region[0].step.y * region[0].readmax
		bufs:insert(buf:subBuffer{
			start = start,
			count = fin - start,
		})
	end
	return bufs
end

local subBuffersForBuffer = table()
local function getSubBuffers(buf)
	local sbufs = subBuffersForBuffer[buf]
	if not sbufs then
		sbufs = chopUpBuffer(buf)
		subBuffersForBuffer[buf] = sbufs
	end
	return sbufs
end


local bInitKernel = program:kernel'bInit'
local bs = getSubBuffers(b)
for i,cmd in ipairs(env.cmds) do
	--bInitKernel.obj:setArgs(regionBufs[i], bs[i])
	bInitKernel.obj:setArg(0, regionBufs[i])
	bInitKernel.obj:setArg(1, bs[i])
	cmd:enqueueNDRangeKernel{
		kernel = bInitKernel.obj,
		dim = env.dim,
		-- kernel domain is based on the write range
		-- while the buffer is based on the read range
		globalSize = domains[i].globalSize.s,
		localSize = domains[i].localSize.s,
	}
end
for i,cmd in ipairs(env.cmds) do
	cmd:finish()
end

local event = require 'cl.event'()
local AKernel = program:kernel'A'


local function A(Y,X)
	-- pick the subbuffers for the buffer ... to chop up the write
	-- the read will still be a solid buffer (right?)
	-- or chop up them both?
	local Xs = getSubBuffers(X)
	local Ys = getSubBuffers(Y)

	for i,cmd in ipairs(env.cmds) do
		AKernel.obj:setArgs(regionBufs[i], Ys[i], Xs[i])
		cmd:enqueueNDRangeKernel{
			kernel = AKernel.obj,
			dim = env.dim,
			globalSize = domains[i].globalSize.s,
			localSize = domains[i].localSize.s,
			event = event,
		}
	end
	for i,cmd in ipairs(env.cmds) do
		cmd:finish()
	end
end

local function splot(gpubuf, name)
	local cpubuf = gpubuf:toCPU()
	gnuplot{
		output = 'krylov-cl-'..name..'.png',
		style = 'data lines',
		griddata = {
			x = matrix{n}:lambda(function(i) return i end),
			y = matrix{n}:lambda(function(i) return i end),
			matrix{n,n}:lambda(function(i,j) return cpubuf[(i-1) + n * (j-1)] end),
		},
		{splot=true, using='1:2:3', title=name},
	}
end

splot(b, 'b')

for _,solver in ipairs{
	--'conjgrad',
	'conjres',
	--'bicgstab',
	--'gmres',
} do
	require('solver.cl.'..solver){
		env = env,
		A = A,
		b = b,
		x = x,
		errorCallback = function(res, iter, x_)
			print(iter, res)
		end,
		restart = 10,
		maxiter = 1000,
	}()
	splot(x, 'x-'..solver)
end

for _,cmd in ipairs(env.cmds) do
	cmd:finish()
end
local start = event:getProfilingInfo'CL_PROFILING_COMMAND_START'
local fin = event:getProfilingInfo'CL_PROFILING_COMMAND_END'
print('duration', tonumber(fin - start)..' ns')
