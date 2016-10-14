local C = {}

function C.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-useGPU', 1, 'Run the test on a GPU (0 is false)')
  cmd:option('-usecudnn', false, 'Enable cudnn')
  cmd:option('-demo', true, 'Show the results from 10 random images on MPII validation')
  cmd:option('-eval', false, 'Evaluate the model')
  cmd:option('-dataset', 'mpii', 'Select the dataset to use: LSP/MPII')
  cmd:option('-res', 256, 'Input resolution')
  cmd:text()
  
  local opt = cmd:parse(arg or {})
  
  opt.dataset:lower()
  
  assert(opt.dataset=='lsp' or opt.dataset=='mpii',"Only mpii and lsp are valid options")
  
  if opt.useGPU<1 then
        opt.useGPU = false
  else
        opt.useGPU = true
  end
  return opt
end

return C
