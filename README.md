# Graph Neural Multi-Behavior Enhanced Recommendation

This repository contains TensorFlow codes and datasets for the paper:

>Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Mengyin Lu, Liefeng Bo (2021). Multi-Behavior Enhanced Recommendation with Cross-Interaction Collaborative Relation Modeling, <a href='https://ieeexplore.ieee.org/abstract/document/9458929'> Paper in IEEE</a>, <a href='https://arxiv.org/abs/2201.02307'> Paper in ArXiv</a>. In ICDE'21, Online, April 19-22, 2021.

## Introduction
Graph Neural Multi-Behavior Enhanced Recommendation (GNMR) incorporates multi-behavior user interaction data to enhance collaborative filtering. It models the behavior heterogeneity and the inter-dependencies between behavior types under a graph-based message passing scheme.


## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{xia2021gnmr,
  author    = {Xia, Lianghao and
               Huang, Chao and
	       Xu, Yong and
	       Dai, Peng and
	       Lu, Mengyin and
	       Bo, Liefeng},
  title     = {Multi-Behavior Enhanced Recommendation with Cross-Interaction Collaborative Relation Modeling},
  booktitle = {2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  year      = {2021},
}
```

## Environment
The codes of GNMR are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
We utilized three datasets to evaluate GNMR: <i>Yelp, MovieLens, </i>and <i>Taobao</i>. The <i>like</i> behavior is taken as the target behavior for Yelp and MovieLens data. The <i>purchase</i> behavior is taken as the target behavior for Taobao data. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions. Except from predicting the target behavior, this repository also includes datasets for prediction overall user-item interactions. In specific, the testing datasets in `Datasets/dataset_name/click` directory were composed by selecting users' last interaction, without consideration of interaction types.

## How to Run the Codes
Please unzip the datasets in `Datasets/` first. Also you need to create the `History/` and the `Models/` directories. The command to train GNMR on the Yelp/MovieLens/Taobao dataset is as follows. The commands specify the hyperparameter settings that generate the reported results in the paper. For overall prediction on Taobao data, we conducted sub-graph sampling to efficiently handle the large-scale multi-behavior user-item graphs.

* Yelp-Target
```
python .\labcode.py --data yelp --target buy --reg 1e-1
```
* Yelp-Overall
```
python .\labcode.py --data yelp --target click
```
* MovieLens-Target
```
python .\labcode.py --data ml10m --target buy --sampNum 80 --epoch 200
```
* MovieLens-Overall
```
python .\labcode.py --data ml10m --target click --sampNum 80 --epoch 200
```
* Taobao-Target
```
python .\labcode.py --data ECommerce --target buy --reg 1
```
Important arguments:
* `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3}`.
* `sampNum`: This parameter denotes the number of training samples for each user. It is tuned between `40` and `80`.


## Acknowledgements
We thank the anonymous reviewers for their constructive feedback and comments. This work is supported by National Nature Science Foundation of China (62072188, 61672241), Natural Science Foundation of Guangdong Province (2016A030308013), Science and Technology Program of Guangdong Province (2019A050510010).


