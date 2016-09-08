# Human pose estimation via Convolutional Part Heatmap Regression

This repository implements a demo of the Human pose estimation via Convolutional Part Heatmap Regression paper [Bulat](https://www.adrianbulat.com)&[Tzimiropoulos](http://www.cs.nott.ac.uk/~pszyt/).

## Requirement
- Install the latest [Torch](http://torch.ch/docs/getting-started.html) version
- Install python 2.7 using the package manager

### Torch packages
- [nn](https://github.com/torch/nn)
- [cunn](https://github.com/torch/cunn) or [cudnn](https://github.com/soumith/cudnn.torch) (preffered) if you have a CUDA enabled GPU
- [xlua](https://github.com/torch/xlua)
- [image](https://github.com/torch/image)
- [sh](https://github.com/zserge/luash)
- [fb.python](https://github.com/facebook/fblualib/blob/master/fblualib/python/README.md)

Most of the listed package can be installed by simple running 
```bash
luarocks install [packagename]
```
For sh and fb.python packages please visit their github repositories and carrefully follow the instruction provided by their authors. 

### Python packages
- [numpy](http://www.scipy.org/scipylib/download.html)
- [matplotlib](http://matplotlib.org/users/installing.html) - required for plotting 

## Trained models
By default, on the first run the scripts will attempt to automatically download the models, however for your convinience they are provided also for separate usage.

| Dataset used  | LSP error   | MPII  error |
| ------------- | ----------- | ----------- |
| [MPII](https://www.adrianbulat.com/downloads/ECCV16/human_pose_mpii.t7)          | -           | 89.7        |
| [MPII + LSP](https://www.adrianbulat.com/downloads/ECCV16/human_pose_lsp.t7)    | 90.7        | -           |

## Usage

The provided code comes along with a series of options. In order to list them please run:
```bash
th main.lua --help
```
To run a demo on 10 random images:
```bash
th main.lua -dataset lsp 
```

To evaluate the model on the validation set for LSP/MPII:
```bash
th main.lua -dataset lsp -eval
```

If you have installed cudnn4 or cudnn5 you can run the demo faster:
```bash
th main.lua -dataset lsp -eval -usecudnn
```

The demo doesn't require a GPU, however having one will speed up the process.

## Notes

For more details/questions please visit the [project page](https://www.adrianbulat.com) or send an email at adrian.bulat@nottigham.ac.uk

Warning: The script will download by default both the models and the dataset(~15Gb), if you wan't to avoid this or you already have them downloaded please move them in the corresponding folders 
in datasets/[datasetname]_dataset/. Running the demo for lsp dataset will require only ~700Mb of space.



