require 'nn'

local DCTConvolution, parent = torch.class('nn.DCTConvolution', 'nn.SpatialConvolution')

function DCTConvolution:__init(inverse)
	parent.__init(self, 1, 64, 8, 8, 8, 8, 0, 0)
	self.inverse = inverse or false
	parent.noBias(self)
	self:reset()
end

local function C(u)
	if u == 1 then
		return 1. / torch.sqrt(2.)
	else
		return 1.
	end
end

local function IDCT_matrix()
	local weight = torch.zeros(64,1,8,8)
	for u=1,8 do
		for v=1,8 do
			for x = 1,8 do
				for y = 1,8 do
					weight[{{(x - 1) * 8 + y}, {1}, {u}, {v}}]:add(0.25 * C(u) * C(v)
				 * torch.cos((2 * x - 1) * (u - 1) * math.pi / 16) * torch.cos((2 * y - 1) * (v - 1) * math.pi / 16))
				end
			end
		end
	end
	return weight
end

local function DCT_matrix()
	local weight = torch.zeros(64,1,8,8)
	for u=1,8 do
		for v=1,8 do
			for x = 1,8 do
				for y = 1,8 do
					weight[{{(u - 1) * 8 + v}, {1}, {x}, {y}}]:add(0.25 * C(u) * C(v)
				 * torch.cos((2 * x - 1) * (u - 1) * math.pi / 16) * torch.cos((2 * y - 1) * (v - 1) * math.pi / 16))
				end
			end
		end
	end
	return weight
end

function DCTConvolution:reset()
	if self.inverse then
		self.weight = IDCT_matrix()
	else
		self.weight = DCT_matrix()
	end
	return
end

-- function DCTConvolution:updateOutput(input)
-- 	self.output = parent.updateOutput(self, input)
-- 	return self.output
-- end

function DCTConvolution:accGradParameters(input, gradOutput, scale)
	return
end

function DCTConvolution:__tostring__()
	s = parent.__tostring__(self)
	if self.inverse then
		return s..' <fixed IDCT> '
	else
		return s..' <fixed DCT> '
	end
end

local DCTReshape, reshape_parent = torch.class('nn.DCTReshape', 'nn.DCTConvolution')

function DCTReshape:updateOutput(input)
	self.orig_output = reshape_parent.updateOutput(self, input)
	local nDim = input:dim()
	-- [nbatch *] 64 * H/8 * W/8
	if nDim == 3 then
		self.transposed_output = self.output:permute(2, 3, 1)
		self.output = self.transposed_output:reshape(torch.LongStorage({1, self.output:size(2), self.output:size(3), 8, 8}))
		-- print(self.output:size())
		-- print(self.output[{{},{6},{6},{},{}}])
		return self.output
	elseif nDim == 4 then
		self.transposed_output = self.output:permute(1, 3, 4, 2)
		self.output = self.transposed_output:reshape(torch.LongStorage({self.output:size(1), 1, self.output:size(3), self.output:size(4), 8, 8}))
		-- print(self.output)
		return self.output
	else 
		error(('DCTReshape only accepts input with 3 or 4 dimensions, got input with %d dimensions'):format(nDim))
	end
end

function DCTReshape:updateGradInput(input, gradOutput)
	local nDim = input:dim()
	if nDim == 3 then
		local reshaped_gradOutput = gradOutput:reshape(self.transposed_output:size())
		local transposed_gradOutput = reshaped_gradOutput:permute(3, 1, 2)
		return reshape_parent.updateGradInput(self, input, transposed_gradOutput)
	elseif nDim == 4 then
		local reshaped_gradOutput = gradOutput:reshape(self.transposed_output:size())
		local transposed_gradOutput = reshaped_gradOutput:permute(1, 4, 2, 3)
		return reshape_parent.updateGradInput(self, input, transposed_gradOutput)
	end
end


function DCTReshape:clearState()
	reshape_parent.clearState(self)
	self.orig_output = nil
	self.transposed_output = nil
	return
end


local SpatialConcat, spatialmap_parent = torch.class('nn.SpatialConcat', 'nn.Module')

-- Expects input to be [nbatch * ] * f * H/h * W/w * h * w
function SpatialConcat:__init()
	spatialmap_parent.__init(self)
end

function SpatialConcat:updateOutput(input)
	assert(input:dim() == 5 or input:dim() == 6, ('spatial map only accepts input with 4 or 5 dimensions, got input with %d dimensions'):format(input:dim()))
	self.output = input:transpose(input:dim() - 2, input:dim() - 1)
	self.size_cached = self.output:size()
	if input:dim() == 5 then
		self.output = self.output:reshape(self.output:size(1), self.output:size(2) * self.output:size(3), 
			self.output:size(4) * self.output:size(5))
	elseif input:dim() == 6 then 
		self.output = self.output:reshape(self.output:size(1), self.output:size(2), self.output:size(3) * self.output:size(4), 
			self.output:size(5) * self.output:size(6))
	end
	return self.output
end

function SpatialConcat:updateGradInput(input, gradOutput)
	local intermediate_gradInput = gradOutput:resize(self.size_cached)
	self.gradInput = intermediate_gradInput:transpose(input:dim() - 2, input:dim() - 1)
	return self.gradInput
end


local JPEGDrop, jpeg_drop_parent = torch.class('nn.dropJpeg', 'nn.Module')

function JPEGDrop:__init(q)
	assert(q and q >= 0 and q < 100)
	self.q = q -- TODO: Add different dropout scheme
end

function JPEGDrop:updateOutput(input)
	self.output = input:clone()
	self.output:narrow(input:dim(), self.q, 9 - self.q):fill(0)
	self.output:narrow(input:dim() - 1, self.q, 9 - self.q):fill(0)
	return self.output
end

function JPEGDrop:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:clone()
	self.gradInput:narrow(gradOutput:dim(), self.q, 9 - self.q):fill(0)
	self.gradInput:narrow(gradOutput:dim() - 1, self.q, 9 - self.q):fill(0)
	return self.gradInput
end

local JPEG, JPEG_parent = torch.class('nn.JPEG', 'nn.Sequential')

function JPEG:__init(q)
	assert(q)
	JPEG_parent.__init(self)
	self.q = q

	local jpeg_transform = nn.Sequential()
	jpeg_transform:add(nn.DCTReshape())

	if q == 100 then
		jpeg_transform:add(nn.Identity()) 
	else 
		jpeg_transform:add(nn.dropJpeg(q))
	end

	jpeg_transform:add(nn.SpatialConcat()) -- Return to image size
	jpeg_transform:add(nn.DCTReshape(true))
	jpeg_transform:add(nn.SpatialConcat())

	JPEG_parent.add(self, nn.Unsqueeze(1,3)) -- [b] * 1 * f * H * w
	JPEG_parent.add(self, nn.SplitTable(2,4)) -- {[b] * 1 * H * w ^ (f) } 
	JPEG_parent.add(self, nn.MapTable():add(jpeg_transform))
	JPEG_parent.add(self, nn.JoinTable(1, 3))
end

local JPEGNoiseUnit, JPEGNoiseUnit_parent = torch.class('nn.JPEGNoiseUnit', 'nn.Module')
function JPEGNoiseUnit:__init(dropout_rate, cutoff_bound)
	JPEGNoiseUnit_parent.__init(self)
	self.d = dropout_rate
	self.c = cutoff_bound
	self.mask = torch.Tensor()
end

function JPEGNoiseUnit:updateOutput(input)
	self.output = input:clone()
	self.mask:resizeAs(input)
	self.mask:bernoulli(1 - self.d) -- 1 with probability 1 - d
	self.output:narrow(input:dim(), self.c, 9 - self.c):fill(0)
	self.output:narrow(input:dim() - 1, self.c, 9 - self.c):fill(0)
	self.output:cmul(self.mask)
	return self.output
end

function JPEGNoiseUnit:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:clone()
	self.gradInput:narrow(gradOutput:dim(), self.c, 9 - self.c):fill(0)
	self.gradInput:narrow(gradOutput:dim() - 1, self.c, 9 - self.c):fill(0)
	self.gradInput:cmul(self.mask)
	return self.gradInput
end

function JPEGNoiseUnit:clearState()
   if self.mask then
      self.mask:set()
   end
   return JPEGNoiseUnit_parent.clearState(self)
end

local JPEGNoise, JPEGNoise_parent = torch.class('nn.JPEGNoise', 'nn.Sequential')

function JPEGNoise:__init(dropout_rate, cutoff_bound)
	JPEGNoise_parent.__init(self)
	JPEGNoise_parent.add(self, nn.DCTReshape())
	JPEGNoise_parent.add(self, nn.JPEGNoiseUnit(dropout_rate, cutoff_bound))
	JPEGNoise_parent.add(self, nn.SpatialConcat()) -- Return to image size
	JPEGNoise_parent.add(self, nn.DCTReshape(true))
	JPEGNoise_parent.add(self, nn.SpatialConcat())
end

local JPEGU, JPEGU_parent = torch.class('nn.JPEGU', 'nn.Sequential')

function JPEGU:__init(yd, yc, uvd, uvc) -- d: dropout rate, c: cut off bound, y/uv: channel
	assert(yd and yc and uvd and uvc, 'JPEGU needs quality specification for both y layers and uv layers')
	JPEG_parent.__init(self)
	self.yd = yd
	self.yc = yc
	self.uvd = uvd
	self.uvc = uvc

	local noise = nn.ParallelTable()
	noise:add(nn.JPEGNoise(self.yd, self.yc))
	if not opt.small and not opt.grayscale then
		noise:add(nn.JPEGNoise(self.uvd, self.uvc))
		noise:add(nn.JPEGNoise(self.uvd, self.uvc))
	end

	JPEGU_parent.add(self, nn.Unsqueeze(1,3)) -- [b] * 1 * f * H * w
	JPEGU_parent.add(self, nn.SplitTable(2,4)) -- {[b] * 1 * H * w ^ (f) }
	JPEGU_parent.add(self, noise)
	JPEGU_parent.add(self, nn.JoinTable(1, 3))
end

local QMy = torch.Tensor({
{16, 11, 10, 16, 24, 40, 51, 61},
{12, 12, 14, 19, 26, 58, 60, 55},
{14, 13, 16, 24, 40, 57, 69, 56},
{14, 17, 22, 29, 51, 87, 80, 62},
{18, 22, 37, 56, 68, 109, 103, 77},
{24, 35, 55, 64, 81, 104, 113, 92},
{49, 64, 78, 87, 103, 121, 120, 101},
{72, 92, 95, 98, 112, 100, 103, 99}
})

local QMuv = torch.Tensor({
{17, 18, 24, 47, 99, 99, 99, 99},
{18, 21, 26, 66, 99, 99, 99, 99},
{24, 26, 56, 99, 99, 99, 99, 99},
{47, 66, 99, 99, 99, 99, 99, 99},
{99, 99, 99, 99, 99, 99, 99, 99},
{99, 99, 99, 99, 99, 99, 99, 99},
{99, 99, 99, 99, 99, 99, 99, 99},
{99, 99, 99, 99, 99, 99, 99, 99}
})

local function quantization_matrix(y, Q)
	local QM
	if y then 
		QM = QMy
	else
		QM = QMuv
	end
	local S
	if (Q < 50) then
		S = 5000/Q
	else
		S = 200 - 2*Q;
	end
	return (S * QM + 50.) / 100.
end

local function dropout_matrix(y, Q)
	local QM = quantization_matrix(y, Q):clamp(1, 128):log()
	local cap = 1 - QM / 7
	return cap:pow(2):clamp(0.1, 0.9)
end

local JPEGQUnit, JPEGQUnit_parent = torch.class('nn.JPEGQUnit', 'nn.Module')
function JPEGQUnit:__init(y, Q, fix)
	JPEGQUnit_parent.__init(self)
	self.DM = dropout_matrix(y, Q)
	self.unsqueeze = nn.Unsqueeze(1)
	self.fix = fix
end

function JPEGQUnit:updateOutput(input)
	self.output = input:clone()
	local reshape_DM = self.DM:clone()
	while reshape_DM:dim() < input:dim() do
		reshape_DM = self.unsqueeze:forward(reshape_DM) -- NO boardcasting :(
	end
	self.mask = torch.ge(reshape_DM:expandAs(input), torch.rand(input:size()):typeAs(input)):typeAs(input)
	if self.fix then
		self.mask:select(self.mask:dim(), 1):select(self.mask:dim() - 1,1):fill(1)
	end
	self.output:cmul(self.mask)
	-- print(self.output)
	
	-- print(self.output)
	return self.output
end

function JPEGQUnit:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:clone()
	self.gradInput:cmul(self.mask)
	return self.gradInput
end

function JPEGQUnit:clearState()
   if self.mask then
		self.mask:set()
   end
   return JPEGQUnit_parent.clearState(self)
end

local JPEGQNoise, JPEGQNoise_parent = torch.class('nn.JPEGQNoise', 'nn.Sequential')

function JPEGQNoise:__init(y, Q, fix)
	JPEGQNoise_parent.__init(self)
	JPEGQNoise_parent.add(self, nn.DCTReshape())
	JPEGQNoise_parent.add(self, nn.JPEGQUnit(y, Q, fix))
	JPEGQNoise_parent.add(self, nn.SpatialConcat()) -- Return to image size
	JPEGQNoise_parent.add(self, nn.DCTReshape(true))
	JPEGQNoise_parent.add(self, nn.SpatialConcat())
end

local JPEGQ, JPEGQ_parent = torch.class('nn.JPEGQ', 'nn.Sequential')

function JPEGQ:__init(Q, fix) -- d: dropout rate, c: cut off bound, y/uv: channel
	assert(Q, 'JPEGU needs quality specification')
	JPEGQ_parent.__init(self)
	self.Q = Q

	local noise = nn.ParallelTable()
	noise:add(nn.JPEGQNoise(true, Q, fix))
	if not opt.small and not opt.grayscale then
		noise:add(nn.JPEGQNoise(false, Q, fix))
		noise:add(nn.JPEGQNoise(false, Q, fix))
	end

	JPEGQ_parent.add(self, nn.Unsqueeze(1,3)) -- [b] * 1 * f * H * w
	JPEGQ_parent.add(self, nn.SplitTable(2,4)) -- {[b] * 1 * H * w ^ (f) }
	JPEGQ_parent.add(self, noise)
	JPEGQ_parent.add(self, nn.JoinTable(1, 3))
end


