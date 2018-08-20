require 'image'

checkpoint = {}
checkpoint.date = tostring(os.date("%y_%m_%d_%X"))

function correct_yuv(tensor, offset) -- batchsize * 3 * a * b or 3 * a * b

	if tensor:size():size() == 4 then
		copy = tensor:clone():float()
		copy[{{}, {2,3}, {}, {}}]:add(offset)
		return copy
	else
		copy = tensor:clone():float()
		copy[{{2,3}, {}, {}}]:add(offset)
		return copy
	end
end

function checkpoint:initialize(testing)
	self.folder = opt.save..'/'..opt.name
	self.sample_folder = self.folder..'/samples'
	path.mkdir(opt.save)
	self.testing = testing
	if not self.testing then
		path.mkdir(self.folder)
		path.mkdir(self.sample_folder)
		self:samples_initialize()
	end
end

function checkpoint:save(epoch)
	print("saving\n")
	local save_path = self.folder..('/epoch_%03d'):format(epoch)
	local to_be_saved = {}
	to_be_saved["model"] = wrapper:clearState()
	to_be_saved["opt"] = opt
	to_be_saved["tester"] = tester
	to_be_saved["confusion"] = confusion
	to_be_saved["adversary"] = adversary:clearState()
	torch.save(save_path..'.t7', to_be_saved)
	collectgarbage('collect')
	self:sample(epoch)
	self:dump_training_stats()
	print("checkpoint saved to "..save_path..".t7\n")
end

function checkpoint:dump_training_stats()
	local f = assert(io.open(self.folder..'/training_stats.csv', 'w'))
	tester:dump(f)
	f:close()
end

function checkpoint:samples_initialize()
	self.sample_origin = {"train", "train", "train", "dev", "dev", "dev"}
	self.sample_payloads = cast(imageProvider:sample())  -- 6 images
	self.sample_msgs = cast(messageProvider:generate():index(1, torch.range(1, self.sample_payloads:size()[1]):long()))
	for i = 1,(self.sample_payloads:size()[1]) do
		local filename = ("/in_%02d.png"):format(i)
		if opt.yuv then 
			image.save(self.sample_folder..filename, image.yuv2rgb(correct_yuv(self.sample_payloads[i], -0.5)))
		else
			image.save(self.sample_folder..filename, self.sample_payloads[i])
		end
	end
	self.msgs = {}
	self.msgs["orig"] = self.sample_msgs
end

function checkpoint:sample(epoch)
	local output = wrapper:forward({self.sample_payloads, self.sample_msgs})
	local imgs_out = output[1]
	local msgs_out = output[2]
	local imgs_transmitted = wrapper.modules[2].output[2]
	self.msgs[epoch] = msgs_out:float():clone()
	torch.save(self.sample_folder..'/messages_dump.t7', self.msgs)
	local f = assert(io.open(self.sample_folder..'/messages_dump.txt', 'w'))
	for k,v in pairs(self.msgs) do
		f:write(("%s: %s\n"):format(tostring(k), tostring(v)))
	end
	f.close()
	torch.save(self.sample_folder..'/messages_dump.t7', self.msgs)
	for i = 1,self.sample_payloads:size()[1] do
		local filename = ("/out_%02d_%03d.png"):format(i, epoch)
		local filename_t = ("/transmitted_%02d_%03d.png"):format(i, epoch)
		if opt.yuv then 
			image.save(self.sample_folder..filename, image.yuv2rgb(correct_yuv(imgs_out[i], -0.5)))
			image.save(self.sample_folder..filename_t, image.yuv2rgb(correct_yuv(imgs_transmitted[i], -0.5)))
		else
			image.save(self.sample_folder..filename, imgs_out[i])
			image.save(self.sample_folder..filename_t, imgs_transmitted[i])
		end
	end
end

function checkpoint:save_final()
	if not self.testing then
		local g = assert(io.open(self.folder..'/'..'results.txt', 'w'))
		self:write_final_report(g)
	end
	local f = assert(io.open(opt.save..'/'..opt.name..'.txt', 'w'))
	self:write_final_report(f)
	
	return
end

function checkpoint:write_final_report(f)
	f:write("================================\n")
	f:write("date: "..self.date..'\n')
	f:write(opt.name..'\n')
	f:write("dataset:\n")
	f:write(imageProvider:description())
	f:write("performance:\n")
	if tester then
		f:write(tester:analyze_performance(true))
	else
		f:write('No tester\n')
	end
	if confusion then
		local img_c, msg_c =  confusion:report()
		f:write("image_confusion:\n")
		f:write(img_c)
		f:write("message_confusion:\n")
		f:write(msg_c)
	else
		f:write('No confusion record\n')
	end
	f:write("options: {\n")
	for k,v in pairs(opt) do
		f:write(("    %s: %s\n"):format(tostring(k), tostring(v)))
	end
	f:write("}\n")

	f:write("model:\n")
	f:write("encoder: "..encoder_description.."\n")
	f:write("decoder: "..decoder_description.."\n")
	f:write("layers:\n")
	f:write(tostring(wrapper).."\n")

	f:write("criterion:\n")
	f:write(criterion_description.."\n")
	f:write("\n================================\n")
	f:close()
end

return checkpoint
