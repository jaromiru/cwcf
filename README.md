This is a source code for paper *Classification with Costly Features using Deep Reinforcement Learning* wrote by *Jaromír Janisch*, *Tomáš Pevný* and *Viliam Lisý*, available at https://arxiv.org/abs/1711.07364.

**Prerequisites:**
- cuda capable hardware
- ubuntu 16.04
- cuda 8/9
- python 3.6 (numpy, pandas, pytorch)

**Usage:**
- use tools `tools/conv_*.py` to prepare datasets; read the headers of those files
- select a dataset to use and copy corresponding file from `consts-template` to `const.py`
- run `python3.6 main.py`
- the run will create multiple log files
- you can use octave or matlab to analyze them with `tools/debug.m`
- you can also evaluate the agent on the test set with `tools/eval_*.py`
