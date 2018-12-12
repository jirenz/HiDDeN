# HiDDeN: Hiding Data with Deep Networks. 
HiDDeN: Hiding Data with Deep Networks. [arxiv](https://arxiv.org/abs/1807.09937) ECCV 2018  
Jiren Zhu*, Russell Kaplan*, Justin Johnson, Li Fei-Fei  
*: These authors contributed equally

Warning: This repo is still WIP, let us know if you encounter bugs or issues.

## Data
A sample dataset is provided in `data/yuv_coco_debug.t7`. Afte you download train2014 of COCO dataset, you can use `HiDDeN/coco_prep.lua` to generate the training dataset.

## Running models
* With the testing dataset. You can run
```
th main.lua --develop --name test-run --type float
```
, training results will be written to `checkpoints/test-run`.  

* If you have cuda setup. You can run 
```
th main.lua --develop --name <experiment name>
```
 instead, as `--type cuda` is default.

If you have the full dataset, remove the `develop` flag.

## Pretrained models
Coming soon...

## Other implementations
* Pytorch implementation by Ando Khachatryan: [https://github.com/ando-khachatryan/HiDDeN](https://github.com/ando-khachatryan/HiDDeN).
