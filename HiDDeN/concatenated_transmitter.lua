single_transmitters = {
	["dropout0.8"] = {
		["transmissionNoiseType"] = "dropout",
		["transmissionDropout"] = 0.8,
	},
	["cropout0.8"] = {
		["transmissionNoiseType"] = "cropout",
		["transmissionCropout"] = 0.8,
	},
	["crop0.8"] = {
		["transmissionNoiseType"] = "crop",
		["transmissionCropSize"] = 0.8,
	},
	["gaussian2"] = {
		["transmissionNoiseType"] = "gaussian",
		["transmissionGaussianSigma"] = 2,
	},
	["jpegn"] = {
		["transmissionNoiseType"] = "jpegn",
		["transmissionJPEGQuality"] = 90,
	},
}
-- 'dropout0.8_jpegn'
-- 'dropout0.8_cropout0.8'
-- 'dropout0.8_crop0.8'
-- 'dropout0.8_gaussian2'
-- 'jpegn_dropout0.8'
-- 'jpegn_cropout0.8'
-- 'jpegn_crop0.8'
-- 'jpegn_gaussian2'
-- 'cropout0.8_dropout0.8'
-- 'cropout0.8_jpegn'
-- 'cropout0.8_crop0.8'
-- 'cropout0.8_gaussian2'
-- 'crop0.8_dropout0.8'
-- 'crop0.8_jpegn'
-- 'crop0.8_cropout0.8'
-- 'crop0.8_gaussian2'
-- 'gaussian2_dropout0.8'
-- 'gaussian2_jpegn'
-- 'gaussian2_cropout0.8'
-- 'gaussian2_crop0.8'

concatenated_transmitters = {}
for name1, setting1 in pairs(single_transmitters) do
	for name2, setting2 in pairs(single_transmitters) do
		if name1 ~= name2 then
			name = name1..'_'..name2
			concatenated_transmitters[name] = {
				setting1, setting2
			}
			-- print(name)
		end
	end
end

local ConcatenatedTransmitter, parent = torch.class('nn.ConcatenatedTransmitter', 'nn.Sequential')

function ConcatenatedTransmitter:__init(recipe)
	parent.__init(self)
	for index, new_opt in ipairs(concatenated_transmitters[recipe]) do
		select_table = nn.SelectTable(1)
		intermediate_transmitter, _ = assert(dofile('./transmitter.lua')(new_opt))
		in_plus_noise = nn.ConcatTable()
		in_plus_noise:add(select_table)
		in_plus_noise:add(intermediate_transmitter)
		parent.add(self, in_plus_noise)
	end
	parent.add(self, nn.SelectTable(2))
end
