# Ada-PU: A Boosting Algorithm for Positive-Unlabeled Learning

This is a reproducing code for Ada-PU in the paper "A Boosting Algorithm for Positive-Unlabeled Learning".

* ```utils.py``` has implementations of the risk estimator for non-negative PU (nnPU) learning. 
* ```train.py``` is an example code of running the algorithm. 

The four used datasets are:
* ```UNSW-NB15``` a binary classiﬁcation dataset.
* ```CIFAR10``` preprocessed in such a way that artifacts form the P class and living things form the N class.
* ```Breast Cancer``` a binary classification dataset.
* ```Epsilon``` a binary classification text dataset.

## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Requirements：
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![colorama](http://img.shields.io/badge/-0.4.5-eee?style=flat&label=colorama)


## Quick start
You can just run the python file, it will be executed once, and the result will be printed. You can try different parameters before you execute the python file.

```console
username@localhost:~$ python3 /src/run_training.py --gpu [$gpu] --dataset [$dataset] --pu_type [$pu_type] --model_type [$model_type] --layer[$layer] 
```
