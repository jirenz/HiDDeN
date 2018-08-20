confusion = {}

function confusion:initialize()
	self.records = {}

	self.msg_threasholds = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}
	self.msg_confusion = {}
	self:clear_msg()

	self.img_threasholds = {1./256, 1./128, 1./64, 1./32, 1./16, 1./8, 1./4}
	self.img_confusion = {}
	self:clear_img()
end


function confusion:clear_msg()
	self.msg_num_records = 0
	for i, msg_threashold in ipairs(self.msg_threasholds) do
		self.msg_confusion[msg_threashold] = 0.
	end
end
function confusion:clear_img()
	self.img_num_records = 0
	for i, img_threashold in ipairs(self.img_threasholds) do
		self.img_confusion[img_threashold] = 0.
	end
end

function confusion:clear()
	self:clear_msg()
	self:clear_img()
end

function confusion:end_epoch(epoch)
	self.records[epoch] = self:report()
	self:clear()
end

function confusion:new_test_batch_msg(output, correct)
	local msg_out = output[2]:float()
	local msg_in = correct[2]:float()
	for i, msg_threashold in ipairs(self.msg_threasholds) do
		self.msg_confusion[msg_threashold] = self.msg_confusion[msg_threashold] + torch.sum(torch.ge(torch.abs(msg_out - msg_in), msg_threashold))
	end
	local msg_elements = 1
	msg_elements = torch.sum(torch.eq(msg_out, msg_out))
	self.msg_num_records = self.msg_num_records + msg_elements
end

function confusion:new_test_batch_img(output, correct)
	local img_out = output[1]:float()
	local img_in = correct[1]:float()
	for i, img_threashold in ipairs(self.img_threasholds) do
		self.img_confusion[img_threashold] = self.img_confusion[img_threashold] + torch.sum(torch.ge(torch.abs(img_out - img_in), img_threashold))
	end
	local img_elements = 1
	img_elements = torch.sum(torch.eq(img_out, img_out))
	self.img_num_records = self.img_num_records + img_elements
end


function confusion:new_test_batch(output, correct)
	self:new_test_batch_msg(output, correct)
	self:new_test_batch_img(output, correct)
end

function confusion:report_msg(output_format)
	str = ''
	local format = output_format or '%f %f\n'
	for k, v in pairs(self.msg_confusion) do
		str = str..format:format(k, v / self.msg_num_records)
	end
	return str
end

function confusion:report_img(output_format)
	str = ''
	local format = output_format or '%f %f\n'
	for k, v in pairs(self.img_confusion) do
		str = str..format:format(k, v / self.img_num_records)
	end
	return str
end

function confusion:report(output_format)
	return self:report_img(output_format), self:report_msg(output_format)
end

confusion:initialize()
return confusion