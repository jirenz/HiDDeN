-- Like dropout, but with random square crops

require 'nn'

local Crop, Parent = torch.class('nn.Crop', 'nn.Module')

function Crop:__init(p, rescale, r)
	-- Args:
	-- `p`: 0 <= p < 1. Fraction of the image to be cropped (by area). So p = .5 => a random crop with area  "0.5 * area of the input" will be dropped 
	-- `stochasticInference`: bool. If false nothing is cropped at test time, but activations are scaled by 1-p. If true, same as training.
	Parent.__init(self)
	self.p = p or 0.5 
	self.train = true
	self.rescale = rescale or false
	self.r = r
	if self.p >= 1 or self.p < 0 then
		error('<Crop> illegal percentage, must be 0 <= p < 1')
	end
end

function Crop:updateOutput(input)
	-- Assumes input is 4D (nbatch, nfeat, h, w) or 3D (nfeat, h, w). If 4D,
	-- all inputs in the batch get the same crop.
	self.output:resizeAs(input):copy(input)
	self.nDim = input:dim()
	self.h = input:size()[self.nDim - 1]
	self.w = input:size()[self.nDim]
	if self.r then
		self.squareWidth = self.r
	else
		self.squareWidth = torch.floor(torch.sqrt(self.p * self.h * self.w))
	end
	if self.squareWidth > self.h or self.squareWidth > self.w then
		error(string.format('p = %f is too big for a square crop when h=%d and w=%d', self.p, self.h, self.w))
	end
	-- sample one point in the plane to be the top left of the square
	self.squareX = torch.floor(torch.uniform(1, self.w + 1 - self.squareWidth + 1)) -- this is correct, even though the two +1's and lack of parens look weird
	self.squareY = torch.floor(torch.uniform(1, self.h + 1 - self.squareWidth + 1))
	return self.output:narrow(self.nDim - 1, self.squareY, self.squareWidth):narrow(self.nDim, self.squareX, self.squareWidth)
end

function Crop:updateGradInput(input, gradOutput)
	local nDim = input:dim()
	self.gradInput:resizeAs(input):fill(0)
	if self.train then
		self.gradInput:narrow(self.nDim - 1, self.squareY, self.squareWidth):narrow(self.nDim, self.squareX, self.squareWidth):add(gradOutput)
	else
		error('backprop only defined while training')
	end
	return self.gradInput
end

function Crop:__tostring__()
	return string.format('%s(%f)', torch.type(self), self.p)
end


function Crop:clearState()
	return Parent.clearState(self)
end