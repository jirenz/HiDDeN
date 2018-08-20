
--- Decoder must take in a image and return a message
local adversarialModel = nn.Sequential()
--- input dimension batchsize * 3 * 32 * 32
local ConvBNReLU = dofile('./ConvBNReLU.lua')
-- ConvBNReLU(adversarialModel, 3, opt.messageLength)
if opt.small or opt.grayscale then
	ConvBNReLU(adversarialModel, 1, opt.adversaryFeatureDepth)
else
	ConvBNReLU(adversarialModel, 3, opt.adversaryFeatureDepth)
end
for i=1,opt.adversaryConvolutions do
	ConvBNReLU(adversarialModel, opt.adversaryFeatureDepth, opt.adversaryFeatureDepth)
end
-- ConvBNReLU(adversarialModel, opt.adversaryFeatureDepth, opt.messageLength)
if opt.aveMax then
	adversarialModel:add(nn.SpatialAveragePooling(opt.maxPoolWindowSize, opt.maxPoolWindowSize, opt.maxPoolStride, opt.maxPoolStride))
	adversarialModel:add(nn.SpatialMaxPooling((opt.imageSize - opt.maxPoolWindowSize) / opt.maxPoolStride + 1, (opt.imageSize - opt.maxPoolWindowSize) / opt.maxPoolStride + 1))
end
if opt.maxAve then
	adversarialModel:add(nn.SpatialMaxPooling(opt.maxPoolWindowSize, opt.maxPoolWindowSize, opt.maxPoolStride, opt.maxPoolStride))
	adversarialModel:add(nn.SpatialAveragePooling((opt.imageSize - opt.maxPoolWindowSize) / opt.maxPoolStride + 1, (opt.imageSize - opt.maxPoolWindowSize) / opt.maxPoolStride + 1))
end
if not opt.maxAve and not opt.aveMax then
	adversarialModel:add(nn.SpatialAdaptiveAveragePooling(1, 1))
end
--- 1 * 1 * opt.adversaryFeatureDepth
adversarialModel:add(nn.View(opt.adversaryFeatureDepth):setNumInputDims(3))
adversarialModel:add(nn.Linear(opt.adversaryFeatureDepth,2))
-- adversarialModel:add(nn.Sigmoid())
-- 
-- adversarialModel:add(nn.View(opt.adversaryFeatureDepth):setNumInputDims(3))
-- adversarialModel:add(nn.Tanh())
--- output dimension batchsize * opt.adversaryFeatureDepth
local adv_description = "adversary_large"
if opt.aveMax then
	adv_description = "adversary_ave_max"
end
if opt.maxAve then
	adv_description = "adversary_max_ave"
end
return adversarialModel, adv_description