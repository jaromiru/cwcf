This is a source code for AAAI 2019 paper *Classification with Costly Features using Deep Reinforcement Learning* wrote by *Jaromír Janisch*, *Tomáš Pevný* and *Viliam Lisý*, available at https://arxiv.org/abs/1711.07364.

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