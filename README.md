# Ada-PU: A Boosting Algorithm for Positive-Unlabeled Learning

This is a reproducing code for Ada-PU in the paper "A Boosting Algorithm for Positive-Unlabeled Learning".

* ```utils.py``` has implementations of the risk estimator for non-negative PU (nnPU) learning [1]. 
* ```train.py``` is an example code of running the algorithm. 

The four used datasets are:
* ```CIFAR-10``` [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) [2] preprocessed in such a way that artifacts form the P class and living things form the N class.
* ```Epsilon``` [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) is a binary classification text dataset.
* ```UNSW-NB15``` [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a binary classiﬁcation dataset.
* ```Breast Cancer``` [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) is a binary classification dataset.


## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Requirements：
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![colorama](http://img.shields.io/badge/-0.4.5-eee?style=flat&label=colorama)


## Quick start
You can just run the python file, it will be executed once, and the result will be printed. You can try different parameters before you execute the python file.

```
python3 /src/train.py \
--dataset breastcancer \
--seed 1 \
--num_clf 100 \
--nnpu 1 \
--beta 0.1 \
--random 1
```

## Example Results

The errors are measured by zero-one loss.
* Test accuracy of comparing with other algorithms in ```img/pn.png```

![test accuracy of comparing with other algorithms](img/pn.png "test accuracy")

* Training error and test error of comparing with Neural Network in ```img/pu.png```

![training error of comparing with Neural Network](img/pu.png "training error")

## Reproduce
| Dataset | Beta | Accuracy |
| ------------- | ------- | -------- |
| CIFAR-10      | 0.1     | 90.42 |
| Epsilon       | 0.9     | 73.10 |
| UNSW-NB15     | 0.2     | 75.94 |
| Breast Cancer | 0.00125 | 92.11 |

## Reference

[1] Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. 
"Positive-Unlabeled Learning with Non-Negative Risk Estimator." Advances in neural information processing systems. 2017.
[2] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009).
