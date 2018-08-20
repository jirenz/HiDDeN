--- wrapper accepts encoder and decoder objects
--- It combines them into one pipeline and output the picture and the message

-- -- Wrapping encoder and decoder

-- Input format {original image, message}

local wrapper = nn.Sequential()

local blank_plus_encoder = nn.ConcatTable()
blank_plus_encoder:add(nn.SelectTable(1))
blank_plus_encoder:add(encoder)

wrapper:add(blank_plus_encoder)
-- input format {original, changed}

local encoded_plus_transmitted = nn.ConcatTable()
encoded_plus_transmitted:add(nn.SelectTable(2))
encoded_plus_transmitted:add(transmitter)
wrapper:add(encoded_plus_transmitted)

local blank_plus_decoder = nn.ParallelTable()
blank_plus_decoder:add(nn.Identity())
blank_plus_decoder:add(decoder)
wrapper:add(blank_plus_decoder)

--- Reformatting input
return wrapper
