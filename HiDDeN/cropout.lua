-- Like dropout, but with random square crops

require 'nn'
-- require 'cudnn'
-- require 'cunn'

local Cropout, Parent = torch.class('nn.Cropout', 'nn.Module')

function Cropout:__init(p, stochasticInference)
  -- Args:
  -- `p`: 0 <= p < 1. Fraction of the image to be cropped (by area). So p = .5 => a random crop with area  "0.5 * area of the input" will be dropped 
  -- `stochasticInference`: bool. If false nothing is cropped at test time, but activations are scaled by 1-p. If true, same as training.
  Parent.__init(self)
  self.p = p or 0.5 
  self.train = true
  self.stochastic_inference = stochasticInference or false
  if self.p >= 1 or self.p < 0 then
    error('<Cropout> illegal percentage, must be 0 <= p < 1')
  end
  self.noise = torch.Tensor()
end

function Cropout:updateOutput(input)
  -- Assumes input is 4D (nbatch, nfeat, h, w) or 3D (nfeat, h, w). If 4D,
  -- all inputs in the batch get the same crop.
  self.output:resizeAs(input):copy(input)
  if self.p > 0 then
    if self.train or self.stochastic_inference then
      local nDim = input:dim()
      local h = input:size()[nDim - 1]
      local w = input:size()[nDim]
      local squareWidth = torch.floor(torch.sqrt(self.p * h * w))
      if squareWidth > h or squareWidth > w then
        error(string.format('p = %f is too big for a square crop when h=%d and w=%d', p, h, w))
      end
      -- sample one point in the plane to be the top left of the square
      local squareX = torch.floor(torch.uniform(1, w + 1 - squareWidth + 1)) -- this is correct, even though the two +1's and lack of parens look weird
      local squareY = torch.floor(torch.uniform(1, h + 1 - squareWidth + 1))
      self.noise:resizeAs(input):fill(1)
      self.noise:narrow(nDim - 1, squareY, squareWidth):narrow(nDim, squareX, squareWidth):fill(0)
      self.output:cmul(self.noise)
    else
      self.output:mul(1-self.p)
    end
  end
  return self.output
end

function Cropout:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  if self.train then
    if self.p > 0 then
      self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
    end
  else
    error('backprop only defined while training')
  end
  return self.gradInput
end

function Cropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function Cropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end