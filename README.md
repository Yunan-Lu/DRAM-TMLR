# DRAM
DRAM: a framework for predicting label distribution from tie-allowed multi-label ranking via conditional Dirichlet mixtures


## Requirements
python=3.7.6, numpy=1.21.6, pandas=1.3.5, scikit-learn=0.24.2, scipy=1.7.3, pytorch=1.13.0+cpu, functools

## Reproducing
Change the directory to this project and run the following command in terminal.

```Terminal
python demo.py
```

## Usage

Here is a simple example of using DRAM.

## Usage
```python
from dram import DRAM_LP
from sklearn.model_selection import train_test_split
from utils import report, reduce_label_distributions

X, D = load_dataset('sj') # this api should be defined by users
R, Y = reduce_label_distributions(D)
Xr, Xs, Rr, _, Yr, _, _, Ds = train_test_split(X, R, Y, D)

# training DRAM-LP
dramlp = DRAM_LP().fit(Xr, Rr, Yr) # X: feature matrix; R: rank matrix; Y: logical matrix
# show the predictive performance
Dhat = dramlp.predict(Xs)
report(Dhat, Ds)
```

## Paper

```latex
@article{Lu2023PredictingLD,
  author={Lu, Yunan and Li, Weiwei and Li, Huaxiong and Jia, Xiuyi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Predicting Label Distribution From Tie-Allowed Multi-Label Ranking}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
}
```
