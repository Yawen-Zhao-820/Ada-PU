# A Boosting Algorithm for Positive-Unlabeled Learning

This is a reproducing code for AdaPU in the paper "A Boosting Algorithm for Positive-Unlabeled Learning".

* ```utils.py``` has implementations of the risk estimator for non-negative PU (nnPU) learning [1]. 
* ```train.py``` is an example code of running the algorithm. 

The four used datasets are:
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) [2] preprocessed in such a way that artifacts form the P class and living things form the N class.
* [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) [3] is a binary classification text dataset.
* [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) [4] is a binary classiﬁcation dataset.
* [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) [5] is a binary classification dataset.


## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Requirements：
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![Cython](http://img.shields.io/badge/-3.0.3-eee?style=flat&logo=Cython&logoColor=3776AB&label=Cython) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![colorama](http://img.shields.io/badge/-0.4.5-eee?style=flat&label=colorama)


## Quick start
You can just run the python file: ```train.py```, it will be executed once with the default setting, and the result will be printed and saved. You can also try different parameters before you execute the python file.

```
python3 src/train.py \
--dataset breastcancer \
--seed 5 \
--num_estimator 100 \
--beta 0.0001 \
--random 1
```

## Reproduce
| Dataset | Beta | Accuracy |
| ------------- | ------- | -------- |
| CIFAR-10      | 0.1     | 86.21 |
| Epsilon       | 0.2     | 73.05 |
| UNSW-NB15     | 0.1     | 76.62 |
| Breast Cancer | 0.0001   | 96.49 |

## Reference

[1] Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. 
"Positive-Unlabeled Learning with Non-Negative Risk Estimator." Advances in neural information processing systems. 2017.

[2] Krizhevsky, Alex, and Geoffrey Hinton. "Learning multiple layers of features from tiny images." (2009).

[3] Yuan, G. X., Ho, C. H., & Lin, C. J. (2012). An Improved GLMNET for L1-regularized Logistic Regression. Journal of Machine Learning Research, 13, 1999-2030.

[4] Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.

[5] W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.
