# PyTorch version Weighted Prediction Error

A WPE implementation using PyTorch.
Note that this module is aimed at doing experiments using WPE in DNN computational graph,
and we didn't optimize the implementation efficiently,
so maybe the computation is very slow and it uses much memory.

## Install
### Requirements

- Python>=3.6
- pytorch>=1.0: See https://pytorch.org/get-started/locally


### Install PyTorch Version WPE

```bash
pip install git+https://github.com/kamo-naoyuki/pytorch_complex
pip install ${REPOSITORY_ROOT}
```

## Example of DNN training
```bash
cd example
./prepare_REVERB_data.sh <wsjcam0> <REVERB_DATA_OFFICIAL>
source env.sh
./train.py
```

## RESULTS
Comming soon
### Setup
- Kadi: 9bf0b6d8db68be01f7036018ca0cdbea31e05d7b
- Using The chain acoustic-model of REVERB Challege recipe.

## Reference

[1]: Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017; https://pdfs.semanticscholar.org/f156/c1b463ad2b41c65220e09aa40e970be271d8.pdf
