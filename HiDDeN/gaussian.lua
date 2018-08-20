require 'nn'

local Guassian, gaussianParent = torch.class('nn.Gaussian', 'nn.Sequential')
function Guassian:__init(num_features, sigma)
	gaussianParent.__init(self)
	local radius = math.ceil(2 * sigma)
	gaussianParent.add(self, nn.SpatialReplicationPadding(radius, radius, radius, radius))
	gaussianParent.add(self, nn.GaussianConvolution(num_features, sigma, radius))
	-- print(self.num_features, self.sigma, self.radius)
end

local GaussianConvolution, parent = torch.class('nn.GaussianConvolution', 'nn.SpatialConvolution')

function GaussianConvolution:__init(num_features, sigma, radius)
	self.num_features = num_features
	self.sigma = sigma 
	self.radius = radius
	parent.__init(self, num_features, num_features, 2 * radius + 1, 2 * radius + 1, 1, 1, 0, 0)
	parent.noBias(self)
	self:reset()
	-- print(self.num_features, self.sigma, self.radius)
end

function GaussianConvolution:reset()
	self.weight = gaussianMatrix(self.num_features, self.sigma, self.radius)
	-- print(self.weight)
end

function gaussianMatrix(num_features, sigma, radius)
	local weight = torch.zeros(num_features, num_features, 2 * radius + 1, 2 * radius + 1)
	for f=1,num_features do
		for x = 1,(2 * radius + 1) do
			for y = 1,(2 * radius + 1) do
				weight[{{f}, {f}, {x}, {y}}] = math.exp(- (x * x + y * y) / (2 * sigma * sigma))
			end
		end
	end
	weight:mul(1 / torch.sum(weight[{{1}, {1}, {}, {}}]))
	return weight
end

function GaussianConvolution:accGradParameters(input, gradOutput, scale)
	return
end

function GaussianConvolution:__tostring__()
	s = parent.__tostring__(self)
	return s..(' <gaussian: sigma: %f, radius: %f> '):format(self.sigma, self.radius)
end