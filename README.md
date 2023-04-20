
### Deep Learning Final Project 


Community detection across social networks is a problem that has been heavily studied and
a primary area of interest in network science. Often, communities are not static in nature, with
large deviation between agent membership across time. Many such works have tried to model
the evolution of networks, but these are often heuristic in nature and/or require current/future
ground truth community memberships. As such, there is still a lack of a general purpose model
that is able to model community evolution without the need for ground truth labels. In this
work, we propose a general dynamic graph neural network based model that has the ability
to identify community structure and track time dependent community evolution while being
trainable end-to-end.


To create the DBLP dataset subset mentioned in the report

download the v-14 version of the DBLP dataset from [here](https://www.aminer.org/citation) 

TO generete place the data in the root directory and execute

```
python make_dblp.py --datset_dir ./data/
```


We would like to thank the authors of 

[Towards Better Dynamic Graph Learning New Architecture and Unified Library](https://arxiv.org/pdf/2303.13047.pdf)

for making their [code](https://github.com/yule-BUAA/DyGLib) public and easy to reproduce
