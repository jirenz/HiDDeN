
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
cudnn.benchmark = true

opt = lapp[[
  -f, --file                   (default "../../jirenz/steg/checkpoints_params_search/msglen_64/epoch_200.t7")       path to model checkpoint .t7 file
  -b, --batchSize              (default 1)      batch size
  -m, --messageLength          (default 64)      message length. must match what file was trained on
  -p, --imagePenaltyCoef       (default 1)         imagePenaltyCoefficient for model
  --messagePenaltyCoef         (default 1)         imagePenaltyCoefficient for model
  --imageSize                  (default 128)      size of image
  --type                       (default cuda)     keep this as cuda


  --encoderFeatureDepth        (default 96)                Depth of feature map of encoder
  --decoderFeatureDepth        (default 96)                Depth of feature map of encoder
  --encoderPreMessageConvolution    (default 8)
  --encoderPostMessageConvolution   (default 2)
  --decoderConvolutions             (default 10)
  --maxPoolWindowSize          (default 4)
  --maxPoolStride              (default 2)
  --thin                       (default true)    Dont load all of CoCo

]]
print(opt)

function get_decoder(model)
  return model.modules[2].modules[2]
end

-- opt = {}
-- opt.file = 'trained_checkpoints/msg3_final.t7'
-- opt.batchSize = 8
-- opt.messageLength = 3
-- opt.type = 'cuda'

-- Load everything
print('Loading data...')
cast = dofile('./typecast.lua')
imageProvider = dofile('./coco_provider.lua')
imageProvider:load(false)
imageProvider:setBatchSize(opt.batchSize)
imageProvider:setDataset("dev")
imageProvider:resetBatches()
messageProvider = dofile('./messageprovider.lua')
messageProvider:configure(opt.batchSize, opt.messageLength, false)

print('Loading model...')
checkpoint = torch.load(opt.file)
model = cast(checkpoint.model)
cudnn.convert(model, cudnn):cuda()
model:evaluate()
print('Model loaded.')

decoder = get_decoder(model)
print(decoder)

-- Criterion shenanigans. Again, should be loaded from the .t7 file
criterion, criterion_description = loadfile('./criterion.lua')(opt)

-- Generate the test data
payloads = cast(imageProvider:getBatch())
msgs = cast(messageProvider:generate())

-- print("messages", msgs)

-- -- Do the forward pass:
-- output = model:forward({payloads, msgs})
-- imgs_out = output[1]
-- msgs_out = output[2]
-- loss = criterion:forward(output, {payloads, msgs})
-- print('loss =', loss)
output = model:forward({payloads, msgs})
imgs_out = output[1]:clone()
black_imgs = torch.mul(imgs_out, 0)
image.save('no_noise_image.png', imgs_out[1])
image.save('black_img.png', black_imgs[1])

payload_decoded = decoder:forward(payloads):clone() -- TODO TEST
msgs_out_orig = output[2]:clone()
msgs_out_black = decoder:forward(black_imgs):clone()

noise_levels = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1}
noise_imgs = {}
msgs_out_noises = {}
for i, v in ipairs(noise_levels) do
  noise_img = torch.add(imgs_out, torch.mul(torch.randn(imgs_out:size()):cuda(), v))
  table.insert(noise_imgs, noise_img)
  image.save('noise_img_'..v..'.png', noise_img[1])

  msgs_out_noise = decoder:forward(noise_img):clone()
  table.insert(msgs_out_noises, msgs_out_noise)
end

-- msgs_out_noise = decoder:forward(noise_img):clone()

-- loss = criterion:forward(output, {payloads, msgs})
-- print('loss =', loss)

-- print(imgs_out)
-- print(msgs)
-- print(msgs_out_black)
-- print(msgs_out_orig)

msg_loss_orig = nn.MSECriterion():forward(msgs_out_orig, msgs)
msg_loss_black = nn.MSECriterion():forward(msgs_out_black, msgs)
print('Orig message loss =', msg_loss_orig)
print('Black message loss =', msg_loss_black)

msg_loss_noises = {}
for i, v in ipairs(noise_levels) do
  table.insert(msg_loss_noises, nn.MSECriterion():forward(msgs_out_noises[i], msgs))
  print('Noise ', v,' message loss =', msg_loss_noises[#msg_loss_noises])
end


-- nn.MSECriterion():forward(msgs_out_noise, msgs)
-- print('Noise message loss =', msg_loss_noise)

-- f = assert(io.open('sample_messages.txt', 'w'))
-- for i=1,opt.batchSize do
--     image.save('noise_img_out'..i..'.png', imgs_out[i])
--     image.save('noise_img_in'..i..'.png', payloads[i])
--     f:write("correct: "..tostring(msgs[i]).."\n")
--     f:write("decoded: "..tostring(msgs_out[i]).."\n")
-- end
-- f:close()
















