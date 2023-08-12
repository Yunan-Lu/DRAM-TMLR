import pandas as pd
import numpy as np
from dram import DRAM_LP
from scipy.io import loadmat
from utils import report, reduce_label_distributions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = 'sj' # sj: SJAFFE, 3dfe: SBU-3DFE
data = loadmat('datasets/%s.mat' % dataset)
X, D = data['features'], data['label_distribution']
R, Y = reduce_label_distributions(D)
param = np.load('config.npy', allow_pickle=True).item()
res = pd.DataFrame()

for seed in range(10):
    Xr, Xs, Yr, Ys, Rr, Rs, Dr, Ds = train_test_split(X, Y, R, D, test_size=0.3, random_state=seed)
    scaler = MinMaxScaler().fit(Xr)
    Xr, Xs = scaler.transform(Xr), scaler.transform(Xs)
    Dhat = DRAM_LP(**eval(param[dataset][seed])).fit(Xr, Rr, Yr).predict(Xs)
    res = pd.concat([res, report(Dhat, Ds, out_form='pandas')], axis=0)

print("The average predictive performance:")
print(res.mean(0))