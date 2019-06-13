# PyTorch Weighted Prediction Error

A WPE implementation using PyTorch.

The set of python code in this repository is an example implementation of the DNN-WPE proposed in \[1\].
The WPE implementation (statistics accumulation, filter calculation, etc) here closely follows the one in [nara_wpe](https://github.com/fgnt/nara_wpe) \[2\].
The code here is just a proof of concept of DNN-WPE. Since it is not optimized in terms of computational efficiency, it may be slow. The directory ./example 
contains training of DNN, test of DNN-WPE based on REVERB challenge data.

## Install
### Requirements

- Python>=3.6
- pytorch>=1.0: See https://pytorch.org/get-started/locally


### Install PyTorch Version WPE

```bash
pip install git+https://github.com/kamo-naoyuki/pytorch_complex
pip install git+https://github.com/nttcslab-sp/dnn_wpe
```

## Example of DNN training
```bash
cd example
pip install -r requirements.txt
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

- [1]: Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017; https://pdfs.semanticscholar.org/f156/c1b463ad2b41c65220e09aa40e970be271d8.pdf
- \[2\]: https://github.com/fgnt/nara_wpe
