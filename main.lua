require 'torch'
require 'xlua'
require 'paths'
require 'image'
local py = require 'fb.python'

require 'transform'

-- Load optional libraries
xrequire('cunn')
xrequire('cudnn')

torch.setdefaulttensortype('torch.FloatTensor')

local options = require 'options'
local data = require 'data'

local opts = options.parse(arg)

local activThresh = 0.003

data.checkIntegrity(opts)

-- Load the model
local model = nil
if opts.dataset == 'mpii' then
  model = torch.load('models/human_pose_mpii.t7')
else 
  model = torch.load('models/human_pose_lsp.t7')
end

if opts.useGPU then 
  if opts.usecudnn then
    cudnn.fastest = true
    cudnn.convert(model, cudnn)
  end
  model = model:cuda()
end

model:evaluate()

-- Load the data
if opts.dataset == 'mpii' then
	dataset = torch.load('dataset/mpii_dataset.t7')
else
	dataset = torch.load('dataset/lsp_dataset.t7')
end

valDataset = {}
for i=1,#dataset do
  if dataset[i].type == 0 then
    valDataset[#valDataset+1] = dataset[i]
    
    -- If LSP map points to MPII order
    if opts.dataset == 'lsp' then
      local temp_pts = valDataset[#valDataset].points:clone():view(14,2)
      local pts = torch.zeros(16,2)
      pts[{{1,6},{}}] = temp_pts[{{1,6},{}}]
      pts[{{11,16},{}}] = temp_pts[{{7,12},{}}]
      pts[{{9,10},{}}] = temp_pts[{{13,14},{}}]
      valDataset[#valDataset].points = pts
      valDataset[#valDataset]['headSize'] = torch.dist(pts[{{14},{}}],pts[{{3},{}}])
      valDataset[#valDataset]['image'] = string.format("im%04d.jpg",i)
    end
  end
end

local ids = nil
-- Execute
if opts.demo and not opts.eval then
  ids = torch.randperm(#valDataset)[{{1,10}}]
elseif opts.image ~= '' or opts.eval then
  ids = torch.range(1,#valDataset)
end
local n = ids:size()[1]

-- Set the progress bar
xlua.progress(0,n)

-- Import python libraries and set pairs
py.exec([=[
import numpy as np
import matplotlib.pyplot as plt
pairs = np.array([[1,2], [2,3], [3,7], [4,5], [4,7], [5,6], [7,9], [9,10], [14,9], [11,12], [12,13], [13,9], [14,15], [15,16]])-1
]=])

local predictions = torch.Tensor(n,16,2)

for i=1,n do
  -- Load and prepare the data
  local img = nil
  if opts.dataset == 'mpii' then
    img = image.load('dataset/mpii_dataset/images/'..valDataset[ids[i]].image)
  else
    img = image.load('dataset/lsp_dataset/images/'..valDataset[ids[i]].image)
  end
  local center = (function() if opts.dataset =='mpii' then return valDataset[ids[i]].center else return torch.Tensor({img:size()[3]/2,img:size()[2]/2}) end end)() 
  local scale = (function() if opts.dataset =='mpii' then return valDataset[ids[i]].scale else return 0.89 end end)() 
  local input = crop(img,center,scale,opts.res)
  input = (function() if opts.useGPU then return input:cuda() else return input end end)()
  
  -- Do the forward pass and get the predicitons
  local output = model:forward(input:view(1,3,opts.res,opts.res))
  
  output = applyFn(function (x) return x:clone() end, output)
  local flippedOut = nil
  if opts.useGPU then
        flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res):cuda()))
  else
        flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res)))
  end
  flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
  output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut):float()
	
  output[output:lt(0)] = 0
  xlua.progress(i,n)
  
  local preds_hm, preds_img = getPreds(output[1], center, scale)
  
  if not opts.eval then
  -- Plot the predicted values
  py.exec([=[
plt.imshow(input.swapaxes(0,1).swapaxes(1,2))
for i in range(pairs.shape[0]):
  # plot only the visible joints
   if np.mean(output[pairs[i,0]]) > activThresh and np.mean(output[pairs[i,1]]) > activThresh or dataset=='lsp':
    plt.plot(preds[[pairs[i,0],pairs[i,1]],0],preds[[pairs[i,0],pairs[i,1]],1],linewidth=3.0)
plt.show()
]=],{input=input:float(), preds = preds_hm:view(16,2), activThresh = activThresh, output = output:view(16,opts.res,opts.res), dataset = opts.dataset})
  else
    predictions[{{i},{},{}}] = preds_img
  end
  
  collectgarbage()
end

if opts.eval then
  distance = evaluate(predictions,valDataset)
  calculateMetrics(distance,opts)
end
