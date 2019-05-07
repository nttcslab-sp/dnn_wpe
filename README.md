# DNN based Weighted Prediction Error
## Install 

Requires pytorch>=1.0: See https://pytorch.org/get-started/locally

```bash
python setup.py install
pip install git+https://github.com/kamo-naoyuki/pytorch_complex
```

## How to use

### Prepare dataset
```bash
./data.sh ${REVERB_ROOT} data/
```

### Training

```bash
python train.py data/train
```

## Apply WPE to your data


```bash
python decode.py data/eval/wav.list
```


# Reference

[1]: The DNN based spectrogram estimation for WPE was proposed by Neural network-based spectrum estimation for online WPE dereverberation; K. Kinoshita et al.. 2017; https://pdfs.semanticscholar.org/f156/c1b463ad2b41c65220e09aa40e970be271d8.pdf
