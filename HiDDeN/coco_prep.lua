require 'image'
require 'xlua'

local preprocessor = torch.class 'Preprocessor'

opt = lapp[[
    -t, --trainSize          (default 10000)                      Number of images in training set
    -d, --devSize            (default 1000)                       Number of images in dev set
    -T, --testSize           (default 1000)                       Number of images in test set
    -D, --debugSize          (default 100)                        Number of images in debug set
    -p, --path               (default "data/train2014")           Path of all images
    -S, --savePath           (default "data/yuv_")                Path for saving t7 files

    -r, --noRandom                                                Start from the beginning instead of taking random sample
    -s, --seed               (default 1234)                       Manual random seed

    --prefix                 (default "COCO_train2014_000000")    Image name prefix
    --suffix                 (default ".jpg")                     Image name suffix

    -w, --width              (default 128)                        Width in pixels, we assume width = height here
    --channels               (default 3)                          Number of channels

    -m, --mode               (default "yuv")                      Color space mode
]]
-- we assume 6 digit padding here


print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)

function Preprocessor:__init(full)
    print("Initializing preprocessor\n")
    local total_size = opt.trainSize + opt.devSize + opt.testSize + opt.debugSize
    local alldata = torch.Tensor(total_size, opt.channels, opt.width, opt.width)

    -- download dataset 
    print("Setting up filenames")
    count = 0
    -- populate filenames
    filenames = {}
    for f in paths.files(opt.path) do
        count = count + 1
        filenames[count] = f
    end

    print(string.format("Found %d files in %s\n", count, opt.path))
    file_index = torch.randperm(count)[{{1,total_size}}]

    -- load dataset
    print("Loading images\n")
    for index=1,total_size do
        local filename = filenames[file_index[index]]
        local img = image.load(opt.path..filename, opt.channels, 'float')
        if img:dim() == 2 then
            require 'nn'
            img = nn.Unsqueeze(1):forward(img)
        end
        if opt.mode == 'yuv' then
            alldata[index] = image.scale(alldata[index], image.rgb2yuv(img))
            alldata[index][2]:add(0.5)
            alldata[index][3]:add(0.5)
        else 
            alldata[index] = image.scale(alldata[index], img)
        end
        xlua.progress(index, total_size)
    end

    local slice_index = 1
    self.train = alldata[{{slice_index,slice_index + opt.trainSize - 1}, {}, {}, {}}]:clone()
    slice_index = slice_index + opt.trainSize
    
    self.dev =  alldata[{{slice_index,slice_index + opt.devSize - 1}, {}, {}, {}}]:clone()
    slice_index = slice_index + opt.devSize
    
    self.test =  alldata[{{slice_index,slice_index + opt.testSize - 1}, {}, {}, {}}]:clone()
    slice_index = slice_index + opt.testSize
    
    self.debug = alldata[{{slice_index,slice_index + opt.debugSize - 1}, {}, {}, {}}]:clone()
    slice_index = slice_index + opt.debugSize
end


preprocesser = Preprocessor()
torch.save(opt.savePath..'coco_train.t7', preprocesser.train)
torch.save(opt.savePath..'coco_dev.t7', preprocesser.dev)
torch.save(opt.savePath..'coco_test.t7', preprocesser.test)
torch.save(opt.savePath..'coco_debug.t7', preprocesser.debug)
f = assert(io.open(opt.savePath..'coco_dataset_info.txt', 'w'))
f:write(tostring(os.date("%y_%m_%d_%X"))..'\n')
f:write("options: {\n")
    for k,v in pairs(opt) do
        f:write(("    %s: %s\n"):format(tostring(k), tostring(v)))
    end
f:write("}\n")
f:close()