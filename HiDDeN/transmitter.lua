-- input format {original, changed}

function transmitter_constructer(opt)
	if opt.transmissionNoiseType == "identity" then
		local transmitter = nn.SelectTable(2)
		return transmitter, "identity"

	elseif opt.transmissionNoiseType == "dropout" then
		local diff = nn.Sequential()
		diff:add(nn.CSubTable())
		diff:add(nn.Dropout(opt.transmissionDropout, true, false, true)) -- Do not rescale, not in place, still dropout in testing
		-- Later use spacial dropout, need modification

		local encoded_plus_diff = nn.ConcatTable()
		encoded_plus_diff:add(nn.SelectTable(2))
		encoded_plus_diff:add(diff)

		local transmitter = nn.Sequential()
		transmitter:add(encoded_plus_diff)
		transmitter:add(nn.CAddTable())
		-- p is the probability that we replace a pixel with the generated pixel
		return transmitter, "dropout"

	elseif opt.transmissionNoiseType == "cropout" then
	    local diff = nn.Sequential()
	    diff:add(nn.CSubTable())
	    diff:add(nn.Cropout(opt.transmissionCropout, true)) -- Still cropout in testing

	    local encoded_plus_diff = nn.ConcatTable()
	    encoded_plus_diff:add(nn.SelectTable(2))
	    encoded_plus_diff:add(diff)

	    local transmitter = nn.Sequential()
	    transmitter:add(encoded_plus_diff)
	    transmitter:add(nn.CAddTable())
	    return transmitter, "cropout"

	elseif opt.transmissionNoiseType == "jpegn" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEGN(opt.transmissionJPEGQuality))

		return transmitter, "JPEG-non-differentiable"

	elseif opt.transmissionNoiseType == "crop" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.Crop(opt.transmissionCropSize))
		-- transmitter:add(nn.SpatialUpSamplingBilinear({oheight=opt.imageSize, owidth=opt.imageSize}))
		return transmitter, "random crop"

	elseif opt.transmissionNoiseType == "jpeg_test" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEG(opt.transmissionJPEGCutoff))
		return transmitter, "differentiable jpeg test"

	elseif opt.transmissionNoiseType == "jpegu" then 
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEGU(opt.transmissionJPEGU_yd, opt.transmissionJPEGU_yc,
								 opt.transmissionJPEGU_uvd, opt.transmissionJPEGU_uvc))
		return transmitter, "differentiable jpeg upgraded"

	elseif opt.transmissionNoiseType == "gaussian" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.Gaussian(3, opt.transmissionGaussianSigma))
		return transmitter, "gaussian"

	elseif opt.transmissionNoiseType == "cropjpeg" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEGU(opt.transmissionJPEGU_yd, opt.transmissionJPEGU_yc,
								 opt.transmissionJPEGU_uvd, opt.transmissionJPEGU_uvc))
		-- transmitter:add(nn.Crop(0.5, 32))
		transmitter:add(nn.Crop(0.5, false, 32))
		return transmitter, "crop+jpeg"

	elseif opt.transmissionNoiseType == "jpegq" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEGQ(opt.transmissionJPEGQuality))
		return transmitter, "jpegq"

	elseif opt.transmissionNoiseType == "jpegq+" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.JPEGQ(opt.transmissionJPEGQuality, true))
		return transmitter, "jpegq+"

	elseif opt.transmissionNoiseType == "combined" then
		local transmitter = nn.CombinedTransmitter(opt.transmissionCombinedRecipe)
		return transmitter, "combined"

	elseif opt.transmissionNoiseType == "cropsize" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.Crop(0.5, false, opt.transmissionCropSize))
		return transmitter, "cropsize"
	elseif opt.transmissionNoiseType == "dropoutcrop" then
		local diff = nn.Sequential()
		diff:add(nn.CSubTable())
		diff:add(nn.Dropout(opt.transmissionDropout, true, false, true)) -- Do not rescale, not in place, still dropout in testing
		-- Later use spacial dropout, need modification

		local encoded_plus_diff = nn.ConcatTable()
		encoded_plus_diff:add(nn.SelectTable(2))
		encoded_plus_diff:add(diff)

		local transmitter = nn.Sequential()
		transmitter:add(encoded_plus_diff)
		transmitter:add(nn.CAddTable())
		-- p is the probability that we replace a pixel with the generated pixel
		transmitter:add(nn.Crop(0.5, false, opt.transmissionCropSize))
		return transmitter, "dropoutcrop"

	elseif opt.transmissionNoiseType == "quantization" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		transmitter:add(nn.Quantization())
		return transmitter, "quantization"

	elseif opt.transmissionNoiseType == "resize" then
		local transmitter = nn.Sequential()
		transmitter:add(nn.SelectTable(2))
		D = opt.transmissionOutsize
		transmitter:add(nn.SpatialUpSamplingBilinear({oheight=D,owidth=D}))
		return transmitter, "resize"

	elseif opt.transmissionNoiseType == 'concatenated' then
		local transmitter = nn.ConcatenatedTransmitter(opt.transmissionConcatenatedRecipe)
		return transmitter, "concatenated"
	else
		print(opt)
		error("No valid transmitter")
	end
end

return transmitter_constructer