

function C(u)
	if u == 1 then
		return 1. / torch.sqrt(2.)
	else
		return 1.
	end
end

r = 8
a = torch.ones(r, r)
-- a[{{1,5}, {}}]:fill(1)
-- a[{{}, {2,4}}]:fill(-1)
-- a[{{2, 4, 6, 8}, {}}]:fill(-1)
b = torch.zeros(r, r)
c = torch.zeros(r, r)
pi = 3.1415926535897932
function DCT(a, b)
	for u=1,r do
		for v = 1,r do
			for x = 1,r do
				for y = 1,r do
					b[{u,v}] = b[{u, v}] + 0.25 * C(u) * C(v) * a[{x, y}]
					 * torch.cos((2 * x - 1) * (u - 1) * pi / 16) * torch.cos((2 * y - 1) * (v - 1) * pi / 16)
				end
			end
		end
	end
end

function IDCT(a, b)
	for u = 1,r do
		for v = 1,r do
			for x = 1,r do
				for y = 1,r do
					b[{x,y}] = b[{x, y}] + 0.25 * C(u) * C(v) * a[{u,v}] * torch.cos((2 * x - 1) * (u - 1) * pi / 16) * torch.cos((2 * y - 1) * (v - 1) * pi / 16)
				end
			end
		end
	end
end




function DCT_COEFF()
end
function IDCT_COEFF()
end
DCT(a, b)
IDCT(b, c)

print('a')
print(a)
print('b')
print(b)
print('c')
print(c)