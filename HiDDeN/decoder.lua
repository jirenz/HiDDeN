--- Decoder must take in a image and return a message

local decoderModel = nn.Sequential()
--- input dimension batchsize * 3 * 32 * 32
local ConvBNReLU = dofile('./ConvBNReLU.lua')
if opt.small or opt.grayscale then
	ConvBNReLU(decoderModel, 1, opt.decoderFeatureDepth)
else
	ConvBNReLU(decoderModel, 3, opt.decoderFeatureDepth)
end
for i=1,opt.decoderConvolutions do
	ConvBNReLU(decoderModel, opt.decoderFeatureDepth, opt.decoderFeatureDepth)
end

ConvBNReLU(decoderModel, opt.decoderFeatureDepth, opt.messageLength)
decoderModel:add(nn.SpatialAdaptiveAveragePooling(1, 1))
decoderModel:add(nn.View(opt.messageLength):setNumInputDims(3))
decoderModel:add(nn.Linear(opt.messageLength,opt.messageLength))
local dec_description = "decoder_large"

return decoderModel, dec_description