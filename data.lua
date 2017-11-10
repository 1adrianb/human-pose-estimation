local sh = require 'sh'

local C = {}

function C.get(url,location)
  local download = sh.command('wget')
  if location ~= nil and not paths.filep(paths.concat(location)) then
    print('Creating directory...')
    paths.mkdir(paths.concat(location))
  end
  download(url..' -P '..paths.concat(location))
end

-- This function tries to download the missing data
function C.checkIntegrity(opts)
  -- Define the required model based on the dataset
  local modelName = nil
  if opts.dataset == 'mpii' then
    modelName = 'human_pose_mpii.t7'
  else
    modelName = 'human_pose_lsp.t7'
  end
  
  -- Check if model already exist
  if not paths.filep(paths.concat('models',modelName)) then
	print('Downloading the pretrained models')
    C.get('https://adrianbulat.com/downloads/ECCV16/'..modelName,'models')
  end
  
  if opts.eval or opts.demo then
    if not paths.dirp(paths.concat('dataset')) then
      paths.mkdir(paths.concat('dataset'))
    end
    if opts.dataset == 'lsp' then
      -- Download LSP dataset
	  if not paths.filep(paths.concat('dataset','lsp_dataset.zip')) and not 
      paths.dirp(paths.concat('dataset/lsp_dataset')) then
		  print('Downloading the LSP dataset...')
		  C.get('http://sam.johnson.io/research/lsp_dataset.zip','dataset')
		  C.get('https://adrianbulat.com/downloads/ECCV16/lsp_dataset.t7','dataset')
		  print('Decompressing dataset...')
		  local cmd = sh.command('unzip')
		  cmd('dataset/lsp_dataset.zip',' -d','dataset/')
	  end
    else
      -- Download MPII dataset
      if not paths.filep(paths.concat('dataset','mpii_human_pose_v1.tar.gz')) and not 
      paths.dirp(paths.concat('dataset/mpii_dataset')) then
	    print('Downloading the MPII dataset...')
        C.get('http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz','dataset')
        C.get('https://adrianbulat.com/downloads/ECCV16/mpii_dataset.t7','dataset')
      end
      
      if not paths.dirp(paths.concat('dataset/mpii_dataset')) then
        paths.mkdir(paths.concat('dataset/mpii_dataset'))
        print('Decompressing dataset (this may take a few minutes)...')
        local cmd = sh.command('tar')
        cmd('-xvf','dataset/mpii_human_pose_v1.tar.gz',' -C','dataset/mpii_dataset/')
      end
    end
  end
  
end

return C
