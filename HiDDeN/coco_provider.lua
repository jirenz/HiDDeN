require 'image'

DataProvider = {}

function DataProvider:description()
	return "coco image:\n color: rgb\n scale: [0,1]\n normalization: false\n size: 3 * 256 * 256\n"
	-- body
end

function DataProvider:load(fixImage, randomImages, grayscale)
	print("Loading training dataset")
	if opt.develop then
		self.train = torch.load('data/yuv_coco_debug.t7')
		self.dev = torch.load('data/yuv_coco_debug.t7')
		self.test = torch.load('data/yuv_coco_debug.t7')
	elseif opt.small then
		if not opt.thin then
			self.train = torch.load('data/yuv_24_grayscale_coco_train.t7')
		end
		self.dev = torch.load('data/yuv_24_grayscale_coco_dev.t7')
		-- self.debug = torch.load('data/yuv_24_grayscale_coco_debug.t7')
	elseif opt.small16 then
		if not opt.thin then
			self.train = torch.load('data/16_grayscale_coco_train.t7')
		end
		self.dev = torch.load('data/16_grayscale_coco_dev.t7')
		self.test = torch.load('data/16_grayscale_coco_test.t7')
		-- self.debug = torch.load('data/16_grayscale_coco_debug.t7')
	else
		-- Default loading
		if not opt.thin then
			self.train = torch.load('data/yuv_coco_train.t7')
		end
		self.dev = torch.load('data/yuv_coco_dev.t7')
		self.test = torch.load('data/yuv_coco_test.t7')
		-- self.debug = torch.load('data/yuv_coco_debug.t7')
	end	

	self.data = {}
	self.indices = {}
	self.currentBatch = {}

	if opt.thin then
		self.data["train"] = self.dev
		self.data["dev"] = self.dev
		self.data["test"] = self.test
	else
		self.data["train"] = self.train
		self.data["dev"] = self.dev
		self.data["test"] = self.test
	end
end

function DataProvider:setBatchSize(size)
	self.batchSize = size
end

function DataProvider:setDataset(dataset)
	self.default_dataset = dataset -- string
end

function DataProvider:getNumBatches(dataset)
	if dataset == nil then
		dataset = self.default_dataset
	end
	return #(self.indices[dataset])
end

function DataProvider:resetBatches(dataset)
	if dataset == nil then
		dataset = self.default_dataset
	end
	self.indices[dataset] = torch.randperm(self.data[dataset]:size(1)):long():split(self.batchSize)
	self.indices[dataset][#(self.indices[dataset])] = nil
	self.currentBatch[dataset] = 1
end

function DataProvider:sample()
	return torch.cat(self:sampleData("train", 3), self:sampleData("dev", 3), 1)
end

function DataProvider:sampleData(dataset, count)
	-- local index = torch.floor(torch.rand(count) * self.data[dataset]:size(1) + 1):long()
	local index = torch.zeros(count):long()
	for i = 1,count do
		index[i] = i
	end
	return self.data[dataset]:index(1, index)
end

function DataProvider:getBatch(dataset)
	if dataset == nil then
		dataset = self.default_dataset
	end
	self.currentBatch[dataset] = self.currentBatch[dataset] + 1
	return self.data[dataset]:index(1, self.indices[dataset][self.currentBatch[dataset] - 1])
end

return DataProvider