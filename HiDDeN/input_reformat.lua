local input_reformat = nn.Sequential()

local inputDispatch = nn.ParallelTable()
inputDispatch:add(nn.Reshape(32 * 32 * 3, true))
inputDispatch:add(nn.Identity())

input_reformat:add(inputDispatch)
input_reformat:add(nn.JoinTable(2))

return input_reformat