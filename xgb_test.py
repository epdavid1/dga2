import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import pywt
import pandas as pd
#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath


iter = np.zeros(400)
for i in range(400):
    start = time.time()
    pickle.load(open('model_xgb3.pkl', 'rb')).predict(pickle.load(open('scaler_noise.pkl', 'rb')).transform([[np.sum(np.square(detail_coeff_noise2)) for detail_coeff_noise2 in pywt.wavedec(pd.read_csv('xgb_data.csv'), wavelet='db4', level=5)]]))
    iter[i] = time.time()-start
print('Average detection time for 400 trials = '+str(int(np.round(np.mean(iter)*1000)))+' ms')