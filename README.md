This is a source code for AAAI 2019 paper *Classification with Costly Features using Deep Reinforcement Learning* wrote by *Jaromír Janisch*, *Tomáš Pevný* and *Viliam Lisý*: [paper](https://jaromiru.com/media/about/aaai19_cwcf_paper.pdf) / [slides](https://jaromiru.com/media/about/aaai19_cwcf_talk.pdf) / [poster](https://jaromiru.com/media/about/aaai19_cwcf_poster.pdf) / [code](https://github.com/jaromiru/cwcf) / [blog](https://jaromiru.com/2019/02/07/hands-on-classification-with-costly-features/)

There is an enhanced version of the article under name *Classification with Costly Features as a Sequential Decision-Making Problem* [paper](https://arxiv.org/abs/1909.02564), which analyzes more settings (hard budget, lagrangian optimization of lambda and missing features). The code is available in the [lagrange](https://github.com/jaromiru/cwcf/tree/lagrange) branch of this repository.

Cite as:
```
@inproceedings{janisch2019classification,
  title={Classification with Costly Features using Deep Reinforcement Learning},
  author={Janisch, Jaromír and Pevný, Tomáš and Lisý, Viliam},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```

**Prerequisites:**
- cuda capable hardware
- ubuntu 16.04
- cuda 8/9
- python 3.6 (numpy, pandas, pytorch 0.4)

**Usage:**
- use tools `tools/conv_*.py` to prepare datasets; read the headers of those files; data is expected to be in `../data`
- pretrained HPC models are in `trained_hpc`, or you can use `tools/hpc_svm.py` to recreate them; they are needed in `../data`
- run `python3.6 main.py --dataset [dataset] --flambda [lambda] --use_hpc [0|1] --pretrain [0|1]`, choose `dataset` from `config_datasets/`
- the run will create multiple log files `run*.dat`
- you can use octave or matlab to analyze them with `tools/debug.m`
- you can also evaluate the agent on the test set with `eval.py --dataset [dataset] --flambda [lambda]`
