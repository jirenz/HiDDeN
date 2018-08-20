-- Apply jpeg compression to tensor. Non-differentiable

require 'image'
require 'nn'

local SaveLoad, Parent = torch.class('nn.SaveLoad', 'nn.Module')

function SaveLoad:__init(yuv, yuv_off)
	-- Args:
	Parent.__init(self)
	self.yuv = yuv or true
	self.yuv_off = yuv_off or -0.5
	self.train = true
	if self.q <= 0 or self.q > 100 then
		error('<JPEGN> illegal quality, must be 1 <= q <= 100')
	end
end

function SaveLoad:updateOutput(input)
	-- Assumes input is 4D (nbatch, 3, h, w) or 3D
	local nDim = input:dim()
	local numChannels, height, width
	assert(nDim == 4 or nDim == 3)
	if nDim == 3 then
		numChannels, height, width = input:size(1), input:size(2), input:size(3)
		input = input:view(1, numChannels, height, width)
	else
		numChannels, height, width  = input:size(2), input:size(3), input:size(4)
	end
	self.output:resizeAs(input)
	self.intermediate = input:float()
	if self.yuv then
		self.intermediate[{{}, {2,3}, {}, {}}]:add(self.yuv_off)
	end
	for i=1,self.intermediate:size(1) do
		if self.yuv then
			rgb = image.yuv2rgb(self.intermediate[{{i},{},{},{}}]:squeeze())
		else
			rgb = self.intermediate[{{i},{},{},{}}]:squeeze()
		end
		if self.yuv then
			decompressed = image.rgb2yuv(decompressed)
			decompressed[{{2,3}, {}, {}}]:csub(self.yuv_off)
			self.output[{{i}, {}, {}, {}}] = decompressed:typeAs(self.output):view(1, numChannels, height, width)
		else
			self.output[{{i}, {}, {}, {}}] = decompressed:typeAs(self.output):view(1, numChannels, height, width)
		end
	end
	if nDim == 3 then
		return self.output:squeeze()
	else
		return self.output
	end
end

function SaveLoad:updateGradInput(input, gradOutput)
	error('JPEGN is Non-differentiable')
	return
end

function SaveLoad:__tostring__()
	 return string.format('%s(%f)', torch.type(self), self.q)
end


function SaveLoad:clearState()
	return Parent.clearState(self)
end