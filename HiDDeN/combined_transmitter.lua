
local transmitter_options = {
	["all_noise"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "dropout",
			["transmissionDropout"] = 0.3,
		},
		{
			["transmissionNoiseType"] = "dropout",
			["transmissionDropout"] = 0.7,
		},
		{
			["transmissionNoiseType"] = "cropout",
			["transmissionCropout"] = 0.3,
		},
		{
			["transmissionNoiseType"] = "cropout",
			["transmissionCropout"] = 0.7,
		},
		{
			["transmissionNoiseType"] = "crop",
			["transmissionCropSize"] = 0.3,
		},
		{
			["transmissionNoiseType"] = "crop",
			["transmissionCropSize"] = 0.7,
		},
		{
			["transmissionNoiseType"] = "gaussian",
			["transmissionGaussianSigma"] = 2,
		},
		{
			["transmissionNoiseType"] = "gaussian",
			["transmissionGaussianSigma"] = 4,
		},
		{
			["transmissionNoiseType"] = "jpegu",
			["transmissionJPEGU_yd"] = 0,
			["transmissionJPEGU_yc"] = 5,
			["transmissionJPEGU_uvd"] = 0,
			["transmissionJPEGU_uvc"] = 3,
		},
		{
			["transmissionNoiseType"] = "jpegq",
			["transmissionJPEGQuality"] = 90,
		},
		-- {
		-- 	["transmissionNoiseType"] = "jpegq",
		-- 	["transmissionJPEGQuality"] = 50,
		-- },
		-- {
		-- 	["transmissionNoiseType"] = "jpegq",
		-- 	["transmissionJPEGQuality"] = 10,
		-- },	
	},
	["crop_drop"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "cropsize",
			["transmissionCropSize"] = 24,
		},
		{
			["transmissionNoiseType"] = "dropoutcrop",
			["transmissionCropSize"] = 24,
			["transmissionDropout"] = 0.6,
		},
	},
	["jpeg_combined"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "jpegq",
			["transmissionJPEGQuality"] = 50,
		},
		-- {
		-- 	["transmissionNoiseType"] = "jpegq",
		-- 	["transmissionJPEGQuality"] = 10,
		-- },
		{
			["transmissionNoiseType"] = "jpegu",
			["transmissionJPEGU_yd"] = 0,
			["transmissionJPEGU_yc"] = 5,
			["transmissionJPEGU_uvd"] = 0,
			["transmissionJPEGU_uvc"] = 3,
		},
	},
	["gaussian_combined"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "gaussian",
			["transmissionGaussianSigma"] = 2,
		},
		{
			["transmissionNoiseType"] = "gaussian",
			["transmissionGaussianSigma"] = 4,
		},
	},
	["crop+drop"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "cropsize",
			["transmissionCropSize"] = 24,
		},
		{
			["transmissionNoiseType"] = "dropout",
			["transmissionDropout"] = 0.5,
		},
	},
	["crop+drop"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "cropsize",
			["transmissionCropSize"] = 24,
		},
		{
			["transmissionNoiseType"] = "dropout",
			["transmissionDropout"] = 0.5,
		},
	},
	["resize"] = {
		{
			["transmissionNoiseType"] = "identity",
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 64,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 80,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 96,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 112,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 144,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 160,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 176,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 192,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 208,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 224,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 240,
		},
		{
			["transmissionNoiseType"] = "resize",
			["transmissionOutsize"] = 256,
		},
	},
}

local CombinedTransmitter, parent = torch.class('nn.CombinedTransmitter', 'nn.Module')

function CombinedTransmitter:__init(recipe)
	parent.__init(self)
	self.transmitters = {}
	self.current = 1
	for index, new_opt in ipairs(transmitter_options[recipe]) do
		self.transmitters[index], _ = assert(dofile('./transmitter.lua')(new_opt))
	end
end

function CombinedTransmitter:updateOutput(input)
	self.output = self.transmitters[self.current]:forward(input)
	return self.output
end

function CombinedTransmitter:updateGradInput(input, gradOutput)
	self.gradInput = self.transmitters[self.current]:backward(input, gradOutput)
	self.current = self.current + 1
	if self.current > #(self.transmitters) then
		self.current = 1
	end
	return self.gradInput
end

function CombinedTransmitter:clearState()
   self.current = 1
   parent.clearState(self)
end
