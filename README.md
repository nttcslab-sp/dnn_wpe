# DNN based Weighted Prediction Error
## Install 

### Install PyTorch Version WPE

The followings are requirements.

- Python>=3.6
- pytorch>=1.0: See https://pytorch.org/get-started/locally


```bash
conda install pytorch cudatoolkit=10.0 -c pytorch
pip install git+https://github.com/kamo-naoyuki/pytorch_complex
pip install ${REPOSITORY_ROOT}
```

### Install modules to run the example
```bash
export PATHON_PATH=${REPOSITORY_ROOT}/example:${PATHON_PATH}
pip install -r example/requirements.txt
```

## How to use

### Prepare REVERB challenge dataset
```bash
./prepare_REVERB_data.sh ${REVERB_CHANLLENGE} data/
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
