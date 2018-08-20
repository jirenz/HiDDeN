
require 'image'
require 'xlua'
require 'optim'
require 'nn'
require './cropout.lua'
require './jpegn.lua'
require './crop.lua'
require './DCT_layer.lua'
require './combined_transmitter.lua'
require './concatenated_transmitter.lua'
require './gaussian.lua'
require './quantization.lua'


opt = lapp[[
	-b, --batchSize              (default 12)                batch size
	-m, --messageLength          (default 30)                  message length (number of floats)
	-i, --imageSize              (default 128)                Image size (3 * imageSize * imageSize)
	-e, --epochs                 (default 200)                number of epochs
	--type                       (default cuda)               cuda/float/cl
	--optimType                  (default adam)               adam or sgd
	--testPer                    (default 1)                  Test every __ epochs
	--savePer                    (default 20)                 Save every __ epochs
	--confusionPer                    (default 20)                 Save every __ epochs

	--loadCheckpoint             (default "")
	--seed                       (default 1234)

	--develop 					 							  Use only debug dataset
	--thin                                                    Use smaller dev dataset
	--small                                                   Use 24 by 24 gray scale dataset
	--small16                                                 Use 16 by 16 gray scale dataset

	--learningRate               (default 1E-3)               Learning Rate
	-p, --imagePenaltyCoef       (default 1)                  Coefficient for image similartiy
	--messagePenaltyCoef         (default 1)                  Coefficient for message correctness
	--save                       (default checkpoints)     path for saving
	--name                       (default "")                 model name

	--encoderFeatureDepth            (default 64)                Depth of feature map of encoder
	--decoderFeatureDepth            (default 64)                Depth of feature map of encoder
	--encoderPreMessageConvolution   (default 3)
	--encoderPostMessageConvolution  (default 1)
	--decoderConvolutions            (default 6)
	--maxPoolWindowSize              (default 4)
	--maxPoolStride                  (default 2) 

	--transmissionNoiseType          (default "identity")         Noise imposed in transmission
	--transmissionDropout            (default 0.4)                Dropout p = probability of changing encoded item into original item
	--transmissionCropout            (default 0.4)                Cropout p = fraction of the image to be cropped, by area
	--transmissionJPEGQuality        (default 50)                 Compression quality for jpeg
	--transmissionCropSize           (default 0.5)                Size of random crop
	--transmissionJPEGCutoff         (default 5)                  Cut off of jpeg frequency domain
	--transmissionJPEGU_yd           (default 0)
	--transmissionJPEGU_yc           (default 5)
	--transmissionJPEGU_uvd          (default 0)
	--transmissionJPEGU_uvc          (default 3)
	--transmissionGaussianSigma      (default 2)
	--transmissionOutsize            (default 128)
	--transmissionCombinedRecipe     (default "")
	--transmissionConcatenatedRecipe (default "")

	-d, --adversary_gradient_scale   (default 0.1)
	--adversaryFeatureDepth      (default 64)
	--adversaryConvolutions      (default 2)

	--fixMessage                                              Fix message content
	--fixImage                                                Fix image content
	--randomImage                                             All images are random noise
	--noSave                                                  Suppress saving
	--noProgress                                              suppress progress bar
	--grayscale
]]


torch.manualSeed(opt.seed)
math.randomseed(opt.seed)

--- message length 64
--- image dimension 3 * opt.imageSize * opt.imageSize
if opt.small or opt.small16 or opt.grayscale then
	opt.yuv = false
	opt.MSEimg = false
	opt.grayscale = true
end


if opt.type == "cuda" then
	require 'cunn'
	require 'cudnn'
	require 'cutorch'
	print('running with cudnn backend')
	cudnn.benchmark = true
end


optimState = {
	learningRate = opt.learningRate,
	learningRateDecay = 0,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1E-8,
}
adv_optimState = {
	learningRate = opt.learningRate,
	learningRateDecay = 0,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1E-8,
}
print(opt)
print(optimState)

cast = assert(dofile('./typecast.lua'))

imageProvider = assert(dofile('./coco_provider.lua'))
imageProvider:load(opt.fixImage, opt.randomImage, opt.grayscale)
imageProvider:setBatchSize(opt.batchSize)

messageProvider = assert(dofile('./messageprovider.lua'))
messageProvider:configure(opt.batchSize, opt.messageLength, opt.fixMessage)

encoder, encoder_description = assert(loadfile('./encoder.lua'))(opt)
decoder, decoder_description = assert(loadfile('./decoder.lua'))(opt)
transmitter, transmitter_description = assert(dofile('./transmitter.lua'))(opt)
adversary, adversary_description = assert(loadfile('./adversary.lua'))(opt)
encoder = cast(encoder)
decoder = cast(decoder)
transmitter = cast(transmitter)
adversary = cast(adversary)
if encoder_description == nil then encoder_description = "encoder" end
if decoder_description == nil then decoder_description = "decoder" end
if transmitter_description == nil then transmitter_description = "transmitter" end
if opt.name == "" then
	opt.name = encoder_description.."+"..transmitter_description.."+"..decoder_description
end

wrapper= assert(loadfile('./wrapper.lua'))(encoder, decoder, transmitter, opt)
if opt.loadCheckpoint and opt.loadCheckpoint ~= "" then
	local ckpt = torch.load(opt.loadCheckpoint)
	wrapper = ckpt.model
end
wrapper = cast(wrapper)
if opt.type == "cuda" then
	cudnn.convert(wrapper, cudnn):cuda()
	cudnn.convert(adversary, cudnn):cuda()
end

-- Load tester
tester = assert(loadfile('./tester.lua'))(opt, wrapper)
-- Load confusion records
confusion = assert(dofile('./confusion.lua'))
--- Checkpoint operates on global vars
checkpoint = dofile('./checkpoint.lua')
checkpoint:initialize()

criterion, criterion_description = assert(loadfile('./criterion.lua'))(opt)
criterion_adv = cast(nn.CrossEntropyCriterion())

adv_params, adv_gradParams = adversary:getParameters()
params, gradParams = wrapper:getParameters()

function train(payloads, msgs)
	local feval = function(x)
		if params ~= x then
			params:copy(x)
		end
		adv_gradParams:zero()
		gradParams:zero()
		local output = wrapper:forward({payloads, msgs})
		local loss = criterion:forward(output, {payloads, msgs})
		local gradOutput = criterion:backward(output, {payloads, msgs})


		-- pred 1 on generated
		local payloads_encoded = output[1]
		local labels_encoded = cast(torch.ones(payloads_encoded:size()[1]))
		local predFake = adversary:forward(payloads_encoded)
		local predFakeLoss = criterion_adv:forward(predFake, labels_encoded)
		local predFakeGradOut = criterion_adv:backward(predFake, labels_encoded)
		local predFakeGradIn = adversary:backward(payloads_encoded, predFakeGradOut)


		local predGeneratorLoss = criterion_adv:forward(predFake, 2 * labels_encoded)
		local predGeneratorGradOut = criterion_adv:backward(predFake, 2 * labels_encoded)
		local predGeneratorGradIn = adversary:backward(payloads_encoded, predGeneratorGradOut, 0)
		gradOutput[1] = gradOutput[1] + opt.adversary_gradient_scale * predGeneratorGradIn

		-- pred 2 on original
		local predTrue = adversary:forward(payloads)
		local predTrueLoss = criterion_adv:forward(predTrue, 2 * labels_encoded)
		local predTrueGradOut = criterion_adv:backward(predTrue, 2 * labels_encoded)
		local predTrueGradIn = adversary:backward(payloads, predTrueGradOut)

		adv_loss = predTrueLoss + predFakeLoss
		
		wrapper:backward({payloads, msgs}, gradOutput)
		return loss, gradParams
	end
	local adv_feval = function (x)
		if adv_data ~= x then 
			adv_params:copy(x)
		end
		return adv_loss, adv_gradParams
	end
	if opt.optimType == 'adam' then
		optim.adam(feval, params, optimState)
		optim.adam(adv_feval, adv_params, adv_optimState)
	else 
		optim.sgd(feval, params, optimState)
		optim.sgd(adv_feval, adv_params, adv_optimState)
	end
end

function train_epoch()
	imageProvider:setDataset("train")
	imageProvider:resetBatches()
	wrapper:training()
	local num_batches = imageProvider:getNumBatches()
	for batch=1,num_batches do
		local payloads = cast(imageProvider:getBatch())
		local msgs = cast(messageProvider:generate())
		train(payloads, msgs)
		if not opt.noProgress then xlua.progress(batch, num_batches) end
	end
end

function test(epoch, do_confusion)
	imageProvider:setDataset("dev")
	imageProvider:resetBatches()
	tester:new_epoch(epoch)
	local num_batches = imageProvider:getNumBatches()
	wrapper:evaluate()
	for batch=1,num_batches do
		local payloads = cast(imageProvider:getBatch())
		local msgs = cast(messageProvider:generate())
		local output = wrapper:forward({payloads, msgs})
		local loss = criterion:forward(output, {payloads, msgs})

		local payloads_encoded = output[1]
		local labels_encoded = cast(torch.ones(payloads_encoded:size()[1]))
		local predFake = adversary:forward(payloads_encoded)
		local predFakeLoss = criterion_adv:forward(predFake, labels_encoded)
		predFake = predFake:clone()

		local predTrue = adversary:forward(payloads)
		local predTrueLoss = criterion_adv:forward(predTrue, 2 * labels_encoded)
		predTrue = predTrue:clone()

		local adv_loss = predFakeLoss + predTrueLoss

		tester:new_test_batch(loss, output, {payloads, msgs}, {adv_loss, predFake, predTrue})
		if do_confusion then
			confusion:new_test_batch(output, {payloads, msgs})
		end
		if not opt.noProgess then xlua.progress(batch, num_batches) end
	end
	tester:end_epoch()
	if do_confusion then
		confusion:end_epoch(epoch)
	end
	print(tester:report_epoch(true))
	if tester:reset_model() then
		wrapper:reset()
	end
end

function main()
	print(opt.name..": ".."starting to train\n")
	for epoch=1,opt.epochs do
		print(string.format("epoch: %d\n", epoch))
		train_epoch()
		local do_test = (epoch % opt.testPer == 0 or epoch == opt.epochs)
		local do_save = (epoch % opt.savePer == 0)
		if do_test then
			test(epoch, epoch == opt.epochs or epoch % opt.confusionPer == 0)
		end
		if do_save and not opt.noSave then
			checkpoint:save(epoch)
		end
	end
	if not opt.noSave then
		checkpoint:save_final()
	end
end

main()

