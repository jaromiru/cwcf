This is an updated source code for paper *Classification with Costly Features as a Sequential Decision-Making Problem* wrote by *Jaromír Janisch*, *Tomáš Pevný* and *Viliam Lisý*: [paper](https://arxiv.org/abs/1909.02564).

This version is enhanced with multiple options, namely:
- lagrangian optimization of lambda
- possibility of choosing an average or hard budget
- working with missing features (nans in the training data)
- reweighting of the dataset

For *Classification with Costly Features using Deep Reinforcement Learning* version, go to the [master](https://github.com/jaromiru/cwcf) branch.

<!-- 
Cite as:
```
@inproceedings{janisch2019classification,
  title={Classification with Costly Features using Deep Reinforcement Learning},
  author={Janisch, Jaromír and Pevný, Tomáš and Lisý, Viliam},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
``` -->

**Prerequisites:**
- cuda capable hardware
- ubuntu 16.04
- cuda 8/9
- python 3.6 (numpy, pandas, pytorch 0.4)

**Usage:**
- use tools `tools/conv_*.py` to prepare datasets; read the headers of those files; data is expected to be in `../data`
- pretrained HPC models are in `trained_hpc`, or you can use `tools/hpc_svm.py` to recreate them; they are needed in `../data`
- run `python3.6 main.py [dataset] [target]`, choose `dataset` from `config_datasets/`
- set `-target_type` to `lambda` or `cost`, the latter automatically finds suitable lambda with lagrangian optimization (see [this article](https://arxiv.org/abs/1909.02564))
- set `-hard_budget` for strict budget per sample (default is average budget)
- run `python3.6 main.py --help` for additional options
- the run will create multiple log files `run*.dat`
- you can use octave or matlab to analyze them with `tools/debug.m`
- you can also evaluate the agent on the test set with `eval.py --dataset [dataset] --flambda [lambda]`