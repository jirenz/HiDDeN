tester = {}

local function NC(a,b)
	local anorms = torch.sqrt(torch.sum(torch.cmul(a,a), 2))
	local bnorms = torch.sqrt(torch.sum(torch.cmul(b,b), 2))
	local ab = torch.sum(torch.cmul(a,b),2)
	local NC = ab:cdiv(anorms):cdiv(bnorms)
	return NC
end

function tester:computeDiff(orig, new, table, threshold)
	local abs = torch.abs(orig - new)
	local abs_mean = torch.mean(abs)
	local abs_max = torch.max(abs)
	local abs_var = torch.var(abs)
	local abs2 = abs:clone()
	local abs_l2 = torch.mean(abs2:cmul(abs2))
	local rel_diff = abs_mean / torch.abs(torch.mean(torch.abs(orig)) + 1E-5)
	if table then
		local count = #table
		table[count + 1] = rel_diff
		table[count + 2] = abs_mean
		table[count + 3] = abs_max
		table[count + 4] = abs_var
		table[count + 5] = abs_l2
		if threshold then
			local BER = torch.sum(torch.ge(abs, threshold)) / torch.numel(abs)
			table[count + 6] = BER
			local NC = torch.mean(NC(orig - 0.5, new - 0.5))
			table[count + 7] = NC
		end
	end
	return rel_diff, abs_mean, abs_max, abs_var, abs_l2
end

function tester:new_epoch(epoch)
	self.current_epoch = epoch
	self.num_batches = 0
	self.results[epoch] = {}
	for i = 1,#(self.metric_index) do
		self.results[self.current_epoch][i] = 0
	end
	return
end

function tester:end_epoch()
	-- print(self.num_batches)
	assert(self.num_batches ~= 0)
	for i = 1,#(self.metric_index) do
		self.results[self.current_epoch][i] = self.results[self.current_epoch][i] / self.num_batches
	end
	return
end

function PSNR( img_in, img_out, channel)
	local range
	-- Y
	if channel == 1 then
		range = 1
	end
	-- U
	if channel == 2 then
		range = 0.436 * 2
	end
	-- V
	if channel == 3 then
		range = 0.615 * 2
	end
	local diff = img_in[{{}, {channel}, {}, {}}] - img_out[{{}, {channel}, {}, {}}]
	local MSE = torch.squeeze(torch.mean(torch.mean(diff:cmul(diff), 3), 4))
	local PSNR = torch.mean(20 * (math.log(range) / math.log(10)) - 10 * (torch.log(MSE) / math.log(10)))
	-- print(('MSE: %f').format(MSE))
	-- print(('PSNR: %f'):format(PSNR))
	return PSNR
end

function tester:new_test_batch(loss, output, correct, adv_data)
	local img_out = output[1]
	local msg_out = output[2]:clamp(0, 1)
	local img_in = correct[1]
	local msg_in = correct[2]
	assert(self.current_epoch)
	local batch_results = {}
	batch_results[1] = self.current_epoch
	batch_results[2] = loss
	self:computeDiff(img_in, img_out, batch_results)
	batch_results[#batch_results + 1] = PSNR(img_in, img_out, 1) --Y
	-- batch_results[#batch_results + 1] = PSNR(img_in, img_out, 2) --U
	-- batch_results[#batch_results + 1] = PSNR(img_in, img_out, 3) --V
	self:computeDiff(msg_in, msg_out, batch_results, self.threshold)

	--- adv_data: {loss, pred_on_encoded, pred_on_original}
	-- print(adv_data[2])
	-- print(torch.ge(adv_data[2][{{}, 1}], adv_data[2][{{}, 2}]))
	local adv_loss = -1
	local adv_tp = -1
	local adv_tn = -1
	if adv_data then
		adv_loss = adv_data[1]
		adv_tp = torch.mean(torch.ge(adv_data[2][{{}, 1}], adv_data[2][{{}, 2}]):float())
		adv_tn = torch.mean(torch.ge(adv_data[3][{{}, 2}], adv_data[3][{{}, 1}]):float())
	end

	batch_results[#batch_results + 1] = adv_loss
	batch_results[#batch_results + 1] = adv_tp
	batch_results[#batch_results + 1] = adv_tn

	for k,v in pairs(batch_results) do
		self.results[self.current_epoch][k] = self.results[self.current_epoch][k] + v
	end
	self.num_batches = self.num_batches + 1
end

function tester:analyze_performance(verbose)
	local response = "average of last %d epochs\n"..self:report(verbose)
	local beg = #(self.results) - 20
	if beg < 1 then
		beg = 1
	end
	output = {}
	total = #(self.results) - beg + 1
	for count = beg,#(self.results) do
		for k,v in pairs(self.results[count]) do
			if output[k] == nil then
				output[k] = 0
			end
			output[k] = output[k] + v / total
		end
	end
	return response:format(total, unpack(output))
end

function tester:report_epoch(verbose, epoch)
	epoch = epoch or self.current_epoch
	local response = self:report(verbose)
	return response:format(unpack(self.results[epoch])).."\n"
end

function tester:report(verbose)
	local response = ""
	for count = 1,#self.metric_index do
		if verbose then
			local format = self.metric_format_v[self.metric_index[count]]
			if (format == nil) then
				format = self.metric_format_v["default"]
			end
			response = response..self.metric_index[count]..format.."\n"
		else 
			local format = self.metric_format_no_v[self.metric_index[count]]
			if (format == nil) then
				format = self.metric_format_no_v["default"]
			end
			response = response..format
		end
	end
	return response
end


function tester:reset_model()
	return false
end

function tester:initialize()
	self.opt = opt
	self.model = wrapper
	self.results = {}
	self.threshold = 0.499
	self:generate_metric_index()
end

function tester:generate_metric_index()
	self.metric_index = {
		"epoch",
		"loss",
		"rel_image_distortion",
		"image_diff_mean",
		"image_diff_max",
		"image_diff_var",
		"img_diff_l2",
		"img_PSNR_Y",
		-- "img_PSNR_U",
		-- "img_PSNR_V",
		"rel_message_distortion",
		"msg_diff_mean",
		"msg_diff_max",
		"msg_diff_var",
		"msg_diff_l2",
		"msg_BER",
		"msg_NC",
		"adv_loss",
		"adv_tp", 
		"adv_tn", 
	}
	self.metric_format_v = {}
	self.metric_format_v["epoch"] = ": %d, "
	self.metric_format_v["default"] = ": %f, "
	
	self.metric_format_no_v = {}
	self.metric_format_no_v["epoch"] = "%d, "
	self.metric_format_no_v["default"] = "%f, "
end

function tester:dump(f)
	for k,v in pairs(self.metric_index) do
		if k ~= 1 then
			f:write(", ")
		end
		f:write(v)
	end
	f:write('\n')
	for rep_epoch = 1,#(self.results) do
		f:write(self:report_epoch(false, rep_epoch))
	end
end

tester:initialize() -- This is bad packaging Needs fixing JZ
return tester