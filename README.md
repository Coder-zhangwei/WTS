# WTS
WTS: A weakly towards strongly supervised learning framework for remote sensing land cover classification using segmentation models

### Introduction
This is the implement of WTS supervised learning framework for remote sensing land cover classification using segmentation models, in which the SRG algorithm is referring to  https://github.com/xtudbxk/DSRG-tensorflow.

### Citing this repository
If you find this code is useful for your research, please consider citing it:
> 
>
> @article{wts,  
> 
>        title={WTS: A weakly towards strongly supervised learning framework for remote sensing land cover classification using segmentation models},
>
>        author={Wei Zhang, Ping Tang, Thomas Corpetti and Lijun Zhao},
>
>        booktitle={Remote Sensing},
>
>        pages={},
>
>        year={2021}
>        }

## Environment
We tested the code on
- keras 2.2
- tensorflow 1.11
- python 3.6

Other dependencies:
- numpy
- tqdm
- gdal
- cv2
- segmentation_models
- matplotlib
- sklearn
- pydensecrf

## Usage
-Train SVM and generate initial seed

python generate_initial_seed.py

-Train Segmentation model and update seed iteratively

python train_update_seed.py


