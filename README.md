# fewshot-CAN
This repository contains the code for the paper:
<br>
[**Cross Attention Network for Few-shot Classification**](https://arxiv.org/pdf/1910.07677.pdf)
<br>
Ruibing Hou, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen
<br>
NeurIPS 2019 
<p align='center'>
  <img src='algorithm.png' width="600px">
</p>

### Abstract

Few-shot classification aims to recognize unlabeled samples from unseen classes given only few labeled samples. The unseen classes and low-data problem make few-shot classification very challenging. Many existing approaches extracted features from labeled and unlabeled samples independently, as a result, the features are not discriminative enough. In this work, we propose a novel Cross Attention
Network to address the challenging problems in few-shot classification. Firstly, Cross Attention Module is introduced to deal with the problem of unseen classes. The module generates cross attention maps for each pair of class feature and query sample feature so as to highlight the target object regions, making the extracted feature more discriminative. Secondly, a transductive inference algorithm is proposed to alleviate the low-data problem, which iteratively utilizes the unlabeled query set to augment the support set, thereby making the class features more representative. Extensive experiments on two benchmarks show our method is a simple, effective and computationally efficient framework and outperforms the state-of-the-arts.
