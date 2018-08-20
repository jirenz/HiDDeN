require 'nn'

local Quantization, Parent = torch.class('nn.Quantization', 'nn.Module')

function Quantization:__init()
	Parent.__init(self)
end

function Quantization:updateOutput(input)
	-- Assumes input is 4D (nbatch, nfeat, h, w) or 3D (nfeat, h, w). If 4D,
	-- all inputs in the batch get the same crop.
	local nDim = input:dim()
	self.output = input:clone():mul(256):int():typeAs(input):mul(1.0/256)
	return self.output
end

function Quantization:updateGradInput(input, gradOutput)
	error('backprop undefined for quantization')
end