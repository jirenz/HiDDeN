local nninit = dofile './nninit_downloaded.lua'

local function ConvBNReLU(model, nInputPlane, nOutputPlane)
  model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):init('weight', nninit.kaiming, {gain = 'relu'}):init('bias', nninit.constant, 0))
  model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  model:add(nn.ReLU(true))
end

return ConvBNReLU