-- encoder must accept image and message and output encoded image

local ConvBNReLU = dofile('./ConvBNReLU.lua')
local nninit = dofile './nninit_downloaded.lua'
local input_feature_depth = 3
-- convolution:add(nn.Identity())
if opt.small or opt.grayscale then
	input_feature_depth = 1
	print("Accepting grayscale input")
else
	input_feature_depth = 3
	print("Accepting non-grayscale input")
end
local convolution = nn.Sequential()
ConvBNReLU(convolution, input_feature_depth, opt.encoderFeatureDepth)
for i=1,opt.encoderPreMessageConvolution do 
	ConvBNReLU(convolution, opt.encoderFeatureDepth, opt.encoderFeatureDepth)
end

local msgPreprocess = nn.Sequential()
msgPreprocess:add(nn.Replicate(opt.imageSize, 3))
msgPreprocess:add(nn.Replicate(opt.imageSize, 4))

local id_plus_convolution = nn.ConcatTable()
id_plus_convolution:add(nn.Identity())
id_plus_convolution:add(convolution)

local inputDispatch = nn.ParallelTable()
inputDispatch:add(id_plus_convolution)
inputDispatch:add(msgPreprocess)
-- output {identity, conv, msg}

-- input {identity, conv, msg}
postProcessConv = nn.Sequential()
postProcessConv:add(nn.NarrowTable(2,2))
postProcessConv:add(nn.JoinTable(1,3))
ConvBNReLU(postProcessConv, opt.encoderFeatureDepth + opt.messageLength, opt.encoderFeatureDepth)
for i=1,opt.encoderPostMessageConvolution do 
	ConvBNReLU(postProcessConv, opt.encoderFeatureDepth, opt.encoderFeatureDepth)
end
-- output img_encoded_with_message

-- input {identity, conv, msg}
local postmsg_plus_id = nn.ConcatTable()
postmsg_plus_id:add(postProcessConv)
postmsg_plus_id:add(nn.SelectTable(1))
-- output {img_encoded_with_message, identity}

-- input {img_encoded_with_message, identity}
postProcessAddIdentity = nn.Sequential()
postProcessAddIdentity:add(nn.JoinTable(1,3))
postProcessAddIdentity:add(nn.SpatialConvolution(opt.encoderFeatureDepth + input_feature_depth, input_feature_depth, 1,1, 1,1, 0,0):init('weight', nninit.kaiming, {gain = 'relu'}):init('bias', nninit.constant, 0))
-- output img_encoded_with_message_convolved_with_identity

local postProcess = nn.Sequential()
postProcess:add(postmsg_plus_id)
postProcess:add(postProcessAddIdentity)

local encoderModel = nn.Sequential()
encoderModel:add(inputDispatch)
encoderModel:add(nn.FlattenTable())
encoderModel:add(postProcess)
--- output dimension batchsize * 3 * 32 * 32
return encoderModel, "encoder_large"