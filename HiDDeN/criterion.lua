require 'nn'

function reshapePayload(orig, tileSize)
	-- orig = N * 1 * H * W
	local N = orig:size(1)
	local H = orig:size(3)
	local W = orig:size(4)
	local hTiles = math.floor(H / tileSize)
	local wTiles = math.floor(W / tileSize)
	local output = orig:clone()
	-- print(output:size())
	output = output:narrow(3, 1, hTiles * tileSize):narrow(4, 1, wTiles * tileSize)
	output = output:reshape(N, 1, hTiles, tileSize, wTiles, tileSize)
	output = output:transpose(4, 5)
	return output:reshape(N * hTiles * wTiles, 1, tileSize, tileSize):clone()
end

function reshapePayloadUndo(out, tileSize)
	local N = 1
	local H = 512
	local W = 512
	local hTiles = math.floor(H / tileSize)
	local wTiles = math.floor(W / tileSize)
	-- print(out:size())
	-- print(N, 1, hTiles, wTiles, tileSize, tileSize)
	local output = out:reshape(N, 1, hTiles, wTiles, tileSize, tileSize):transpose(4, 5)
	-- print(output:size())
	output = output:reshape(N, 1, hTiles * tileSize, wTiles * tileSize)
	-- print(output:size())
	return output
end

cast = dofile('./typecast.lua')
criterion = cast(nn.ParallelCriterion())

image_criterion = cast(nn.MSECriterion())
message_criterion = cast(nn.MSECriterion())
criterion:add(image_criterion, opt.imagePenaltyCoef)
criterion:add(message_criterion, opt.messagePenaltyCoef)
return criterion, "MSE+MSE"