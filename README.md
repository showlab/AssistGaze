[EMNLP23 Poster] # GazeVQA 

This is the official repository which provides a baseline model for our proposed task: GazeVQA: A Video Question Answering Dataset for Multiview Eye-Gaze Task-Oriented Collaborations.

[[Paper]](https://aclanthology.org/2023.emnlp-main.648/)

Model Architecture (see [[Paper]](https://arxiv.org/abs/2203.04203) for details):

![arch](https://github.com/showlab/AssistGaze/blob/main/Architecture.pdf)


## Install
(1) PyTorch. See https://pytorch.org/ for instruction. For example,
```
conda install pytorch torchvision torchtext cudatoolkit=11.3 -c pytorch
```
(2) PyTorch Lightning. See https://www.pytorchlightning.ai/ for instruction. For example,
```
pip install pytorch-lightning
```

## Data
The released dataset is under this repository
[[Dataset]](https://github.com/mfurkanilaslan/GazeVQA)
## Encoding

Before starting, you should encode the instructional videos, scripts, QAs.

## Training 

Select the config file and simply train, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Evaluation

## Contact

Feel free to contact us if you have any problems: chenan.song@u.nus.edu, or leave an issue in this repo.
