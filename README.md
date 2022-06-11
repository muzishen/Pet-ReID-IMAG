# Pet-ReID-IMAG
 The 3rd place solution to CVPR2022 Biometrics Workshop Pet Biometric Challenge
---- 
## Introduction
- We just trained one model (ResNeSt) with different scales (i.e., 224, 256, and 288).
- Traing time cost ~1.5 hour with a V100 16GB, so easy, no bells and whistles!
- Techical details are described in our [arXiv preprint paper](https://arxiv.org/pdf/2205.15934.pdf). 

The data obtained by offline addition can be obtained from [here](https://pan.baidu.com/s/1yYNJFuyrJy8kn5TVA5_Okw) [d0kc].
## Requirements

* PyTorch  1.7.0+cu101
* torchvision  0.8.1+cu101 


## Training instruction
```
pip install -r  requirements.txt; cd fastreid/evaluation/rank_cylib; make all
```
```
bash train1.sh
bash train2.sh
bash train3.sh
bash train4.sh
```


## Test on Pet Challenge
```
bash predict.sh
```

## Acknowledgement
A large portion of code is borrowed from [fast-reid](https://github.com/JDAI-CV/fast-reid), many thanks to their wonderful work!
